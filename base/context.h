/**
 * Copyright 2017 Baidu Inc. All Rights Reserved.
 *
 * \brief Context for different device
 * \file context.h
 * \author zhangshuai(zhangshuai20@baidu.com)
 * \date 2017-10-23
 */
#ifndef ANAKIN2_SABER_CONTEXT_H
#define ANAKIN2_SABER_CONTEXT_H

#include "devices.h"
#include "type.h"
#include "anakin_config.h"
#include "utils/logger/logger.h"

#ifdef USE_CUDA
#include "cuda_error.h"
#endif

namespace anakin{

namespace saber {

//! Context info contains specified device info.
template<TargetType target>
class Context {
public:
    //Context(Device<target>& device):_dev(device) {}
    Context(){}
    virtual ~Context() {}
    inline bool operator==(Context<target> &b){
        return _dev == b;
    }

    inline TargetType get_target_type() {
        return target;
    }

    inline int get_dev_id(){
        return _dev.device_ids[0];
    }

    inline void set_dev_id(int dev_id) {
        _dev.device_ids[0] = dev_id;
    }

    inline void init_context();
    inline void bind_dev();

    //inline std::string get_dev_info() { return _dev.device_info(); }
    const TargetType type = target;
public:
    static Device<target> _dev;

};

#ifdef USE_ARM_PLACE
template <>
class Context<ARM> {
public:
    Context<ARM>() : _power_mode(AK_HIGH_POWER){}
    //Context<ARM>(Device<ARM>& device) : _dev(device){}

    virtual ~Context<ARM>() {}

    inline bool operator==(Context<ARM> &b){
        return _dev == b._dev;
    }

    inline TargetType get_target_type() {
        return ARM;
    }

    inline int get_dev_id(){
        return int(_power_mode);
    }

    inline std::vector<int> get_active_ids(){
        return activate_ids;
    }
    //inline std::vector<int> get_dev_id(){ return _dev.device_ids;}
    inline void set_dev_id(int dev_id){
        _power_mode = PowerMode(dev_id);
        set_power_mode(_power_mode);
    }


    inline void set_dev_id_cores(const std::vector<int>& dev_ids){
        //dev_ids is empty, set to core 0
        if (dev_ids.size() == 0){
            activate_ids.resize(1);
            activate_ids[0] = _dev.device_ids[0];
        }else {
            activate_ids.clear();
            for (int i = 0; i < dev_ids.size(); ++i) {
                if (dev_ids[i] < _dev.device_ids.size()){
                    activate_ids.push_back(dev_ids[i]);
                }
            }
        }
    }

    inline void bind_dev(){
        set_cpu_affinity(activate_ids);
    }

    inline void set_power_mode(PowerMode mode){
        if (mode == AK_FULL_POWER){
            activate_ids = _dev.device_ids;
        }
        else if (mode == AK_LOW_POWER) {
            activate_ids.clear();
            for (int i = 0; i < _dev.cluster_ids.size(); ++i) {
                if (_dev.cluster_ids[i] == 1) {
                    activate_ids.push_back(_dev.device_ids[i]);
                }
            }
            if (activate_ids.size() == 0){
                LOG(WARNING)<<"LOW POWER MODE is not support";
                activate_ids.push_back(_dev.device_ids[0]);
            }
        }
        else if (mode == AK_HIGH_POWER){
            activate_ids.clear();
            for (int i = 0; i < _dev.cluster_ids.size(); ++i) {
                if (_dev.cluster_ids[i] == 0) {
                    activate_ids.push_back(_dev.device_ids[i]);
                }
            }
            if (activate_ids.size() == 0){
                activate_ids.push_back(_dev.device_ids[0]);
            }
        }
    }

    inline void init_context(){
        set_power_mode(_power_mode);
    }
    const TargetType type = ARM;
public:
    static Device<ARM> _dev;

private:
    PowerMode _power_mode;
    std::vector<int> activate_ids;

};
#endif

#ifdef USE_CUDA
template<>
class Context<RTCUDA> {
public:
    //Context(Device<RTCUDA>& device): _dev (device) {}

    Context(int stream_num = 1){
        if (stream_num < 1){

            stream_num = 1;
        }
        _stream_num = stream_num;
        _activate_stream_id = 0;
        _dev_id = 0;
        _flag_init = false;
    }

    virtual ~Context() {}

    inline TargetType get_target_type() {
        return _dev.getTargetType();
    }

    inline int get_dev_id(){
        return _dev_id;
    }

    inline void set_dev_id(int dev_id) {
        if (dev_id > _dev.device_num - 1){
            LOG(WARNING) << "exceed the maximum device number: " << _dev.device_num;
            dev_id = 0;
        }

        _dev_id = dev_id;
    }

    /**
     * \brief use as set device
     */
    inline void bind_dev() {
        int current_device;
        CUDA_CHECK(cudaGetDevice(&current_device));
        if (_dev_id != current_device) {
            CUDA_CHECK(cudaSetDevice(_dev_id));
        }
    }

    inline void init_context() {
        _cuda_stream.resize(_stream_num);
        bind_dev();
        for (int i = 0; i < _stream_num; ++i) {
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            _cuda_stream[i] = stream;
        }
        _flag_init = true;
    }

    inline cudaStream_t cuda_stream(int id = 0) {
        //LOG(INFO) << "context flag: " << _flag_init << "streamid: " << _activate_stream_id;
        CHECK_EQ(_flag_init, true) << "context is not initialized!";
        if (id > _stream_num - 1) {
            LOG(WARNING) << "exceed the maximum number of streams: " \
                << _stream_num;
            id = 0;
        }
        return _cuda_stream[id];
    }

    inline cudaStream_t cuda_activate_stream() {
        CHECK_EQ(_flag_init, true) << "context is not initialized!";
        return _cuda_stream[_activate_stream_id];
    }

    inline void set_stream(int id = 0) {
        if (id > _stream_num - 1) {
            LOG(WARNING) << "exceed the maximum number of streams: " \
                << _stream_num;
            _activate_stream_id = 0;
        } else{
            _activate_stream_id = id;
        }
    }

    const TargetType type = RTCUDA;

private:
    bool _flag_init;

public:
    static Device<RTCUDA> _dev;
    int _dev_id;
    int _stream_num;
    int _activate_stream_id;
    std::vector<cudaStream_t> _cuda_stream;
};
#endif

} //namespace saber

} //namespace anakin
#endif //ANAKIN2_SABER_CONTEXT_H
