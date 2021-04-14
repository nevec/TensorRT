/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "siluPlugin.h"
#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::PluginType;

namespace
{
const char* SILU_PLUGIN_VERSION{"1"};
const char* SILU_PLUGIN_NAME{"Silu"};
} // namespace

PluginFieldCollection SiLUPluginCreator::mFC{};
std::vector<PluginField> SiLUPluginCreator::mPluginAttributes;

int SiLUInference(cudaStream_t stream, int size, float* inputData, float* outputData);

SiLUPlugin::SiLUPlugin() : mBatchDim(0)
{
}

SiLUPlugin::SiLUPlugin(const void* data, size_t length)
{
}

int SiLUPlugin::getNbOutputs() const
{
    return 1;
}

int SiLUPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void SiLUPlugin::terminate() {}

Dims SiLUPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // CHW
    
    nvinfer1::Dims dimsOutput;
    dimsOutput.nbDims = inputs->nbDims;
    return dimsOutput;
}

size_t SiLUPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

size_t SiLUPlugin::getSerializationSize() const
{
    return 0;
}

void SiLUPlugin::serialize(void* buffer) const
{
}

void SiLUPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize)
{
    ASSERT(nbInputs == 1);
    auto type = inputTypes[0];
    ASSERT(type == DataType::kFLOAT && format == PluginFormat::kNCHW);
    ASSERT(mBatchDim == 1);
    ASSERT(nbOutputs == 1);
    for (int i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

bool SiLUPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return type == DataType::kFLOAT;
}

const char* SiLUPlugin::getPluginType() const
{
    return SILU_PLUGIN_NAME;
}

const char* SiLUPlugin::getPluginVersion() const
{
    return SILU_PLUGIN_VERSION;
}

void SiLUPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* SiLUPlugin::clone() const
{
    auto* plugin = new SiLUPlugin();
    return plugin;
}

void SiLUPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* SiLUPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType SiLUPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

bool SiLUPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool SiLUPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

int SiLUPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    int status = SiLUInference(stream, mBatchDim * batchSize, (float*)inputData, (float*)outputData);
    ASSERT(status == 0);
    return status;
}


// Plugin creator
SiLUPluginCreator::SiLUPluginCreator() {}

const char* SiLUPluginCreator::getPluginName() const
{
    return SILU_PLUGIN_NAME;
}

const char* SiLUPluginCreator::getPluginVersion() const
{
    return SILU_PLUGIN_VERSION;
}

const PluginFieldCollection* SiLUPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* SiLUPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    SiLUPlugin* plugin = new SiLUPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* SiLUPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    SiLUPlugin* plugin = new SiLUPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}



// static const char* SiLU_PLUGIN_VERSION{"1"};
// static const char* SiLU_PLUGIN_NAME{"Silu"};

// PluginFieldCollection SiLUPluginCreator::mFC{};
// std::vector<PluginField> SiLUPluginCreator::mPluginAttributes;


// SiLU::SiLU() : mBatchDim(1)
// {
// }

// SiLU::SiLU(const void* buffer, size_t length)
// {
// }

// int SiLU::getNbOutputs() const
// {
//     return 1;
// }

// Dims SiLU::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
// {
//     ASSERT(nbInputDims == 1);
//     ASSERT(index == 0);
//     return inputs[0];
// }



// size_t SiLU::getSerializationSize() const
// {
//     // mNegSlope, mBatchDim
//     return 0;
// }

// void SiLU::serialize(void* buffer) const
// {
// }

// void SiLU::configureWithFormat(
//     const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int)
// {
//     ASSERT(type == DataType::kFLOAT && format == PluginFormat::kNCHW);
//     ASSERT(mBatchDim == 1);
//     ASSERT(nbOutputs == 1);
//     for (int i = 0; i < inputDims[0].nbDims; ++i)
//     {
//         mBatchDim *= inputDims[0].d[i];
//     }
// }

// bool SiLU::supportsFormat(DataType type, PluginFormat format) const
// {
//     return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
// }

// int SiLU::initialize()
// {
//     return 0;
// }

// void SiLU::terminate() {}

// size_t SiLU::getWorkspaceSize(int maxBatchSize) const
// {
//     return 0;
// }

// const char* SiLU::getPluginType() const
// {
//     return SiLU_PLUGIN_NAME;
// }

// const char* SiLU::getPluginVersion() const
// {
//     return SiLU_PLUGIN_VERSION;
// }

// void SiLU::destroy()
// {
//     delete this;
// }

// IPluginV2Ext* SiLU::clone() const
// {
//     IPluginV2Ext* plugin = new SiLU();
//     plugin->setPluginNamespace(mNamespace.c_str());
//     return plugin;
// }

// SiLUPluginCreator::SiLUPluginCreator()
// {
// }

// const char* SiLUPluginCreator::getPluginName() const
// {
//     return SiLU_PLUGIN_NAME;
// }

// const char* SiLUPluginCreator::getPluginVersion() const
// {
//     return SiLU_PLUGIN_VERSION;
// }

// const PluginFieldCollection* SiLUPluginCreator::getFieldNames()
// {
//     return &mFC;
// }

// IPluginV2Ext* SiLUPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
// {
//     IPluginV2Ext* plugin = new SiLU();
//     plugin->setPluginNamespace(mNamespace.c_str());
//     return plugin;
// }

// IPluginV2Ext* SiLUPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
// {
//     // This object will be deleted when the network is destroyed, which will
//     // call SiLUPlugin::destroy()
//     IPluginV2Ext* plugin = new SiLU(serialData, serialLength);
//     plugin->setPluginNamespace(mNamespace.c_str());
//     return plugin;
// }

