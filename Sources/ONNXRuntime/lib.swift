// lib.swift
// Copyright (c) 2025 FrogSquare
// Created by Ramesh (Ravone)

import CONNX
import Foundation

typealias ORT_API = UnsafePointer<OrtApi>?

public enum OrtError: Error {
    case Status(String)
}

@frozen
public struct ONNX {
    var api: UnsafePointer<OrtApi>

    var env: OpaquePointer?
    var session: OpaquePointer?
    var sessionOptions: OpaquePointer?

    var usingGPU: Bool = false

    public var isValid: Bool { env != nil && session != nil }

    init() {
        api = OrtGetApiBase()!.pointee.GetApi(UInt32(ORT_API_VERSION))!
    }

    public mutating func Release(tensor: OrtTensor) {
        assert(tensor.isValid)
        api.pointee.ReleaseValue(tensor.ptr)
    }

    public mutating func Release() {
        if usingGPU, let sessionOptions {
            api.pointee.ReleaseSessionOptions(sessionOptions)
            self.sessionOptions = nil
        }
        if let session {
            api.pointee.ReleaseSession(session)
            self.session = nil
        }
        if let env {
            api.pointee.ReleaseEnv(env)
            self.env = nil
        }
    }

    public mutating func CreateEnv(name: String = "ONNX_BASE") throws(OrtError) {
        if let status: OrtStatusPtr = api.pointee.CreateEnv(ORT_LOGGING_LEVEL_FATAL, name, &env) {
            api.pointee.ReleaseStatus(status)
            let message = String(cString: api.pointee.GetErrorMessage(status)!)
            throw .Status("Cannot create Environment \(message)")
        }
        assert(env != nil)
    }

    public mutating func CreateSession(model path: String, usingGPU gpu: Bool = false) throws(OrtError) {
        assert(env != nil, "Call CreateEnv() before Creating a Session")

        usingGPU = gpu

        var status: OrtStatusPtr?
        var sessionOptions: OpaquePointer?
        if usingGPU {
            status = api.pointee.CreateSessionOptions(&sessionOptions)

            if let status {
                let message = String(cString: api.pointee.GetErrorMessage(status)!)
                throw .Status("Cannot Create Session: \(message)")
            }

            guard sessionOptions != nil else {
                throw .Status("SessionOptions is nil")
            }

            var o = OrtCUDAProviderOptions()
            o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault
            o.gpu_mem_limit = 1024 * 1024 * 1024
            o.tunable_op_enable = 0
            o.tunable_op_tuning_enable = 0
            // o.use_legacy_conv_add_activation = 1

            status = api.pointee.SessionOptionsAppendExecutionProvider_CUDA(
                sessionOptions,
                &o
            )

            if let status {
                let message = String(cString: api.pointee.GetErrorMessage(status)!)
                api.pointee.ReleaseStatus(status)
                throw .Status("Cannot Enable CUDA \(message)")
            }
        }

        status = api.pointee.CreateSession(env, path, sessionOptions, &session)
        if let status {
            let message = String(cString: api.pointee.GetErrorMessage(status)!)

            api.pointee.ReleaseEnv(env)
            api.pointee.ReleaseStatus(status)
            throw .Status("Cannot load model: \(message)")
        }
        assert(session != nil)

        if let sessionOptions {
            _ = api.pointee.DisableMemPattern(sessionOptions)
            _ = api.pointee.SetSessionGraphOptimizationLevel(sessionOptions, ORT_DISABLE_ALL)
        }
    }

    public func CreateInput(
        name: String,
        data: UnsafeMutableRawPointer,
        shape: [Int64]
    ) throws(OrtError) -> OrtTensor {
        assert(env != nil && session != nil)

        var status: OrtStatusPtr?
        var memoryInfo: OpaquePointer?
        var inputTensor: OpaquePointer?

        status = if usingGPU {
            api.pointee.CreateMemoryInfo(
                "Cuda",
                OrtDeviceAllocator,
                0,
                OrtMemTypeDefault,
                &memoryInfo
            )
        } else {
            api.pointee.CreateCpuMemoryInfo(
                OrtArenaAllocator,
                OrtMemTypeDefault,
                &memoryInfo
            )
        }

        if let status {
            let message = String(cString: api.pointee.GetErrorMessage(status)!)
            api.pointee.ReleaseStatus(status)
            throw .Status("Cannot Create MemoryInfo \(message)")
        }

        let size = Int(shape[0] * shape[1] * (shape[2] * shape[3]))
        status = api.pointee.CreateTensorWithDataAsOrtValue(
            memoryInfo,
            data,
            size * MemoryLayout<Float>.size,
            shape,
            shape.count,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &inputTensor
        )

        if let status {
            let message = String(cString: api.pointee.GetErrorMessage(status)!)
            api.pointee.ReleaseStatus(status)
            throw .Status("Cannot create Input Tensor \(message)")
        }
        api.pointee.ReleaseMemoryInfo(memoryInfo)

        return OrtTensor(name: name, ptr: inputTensor)
    }

    public func Run(
        withInputs inputs: [OrtTensor],
        outputNames: [String]
    ) throws(OrtError) -> [String: OrtTensor] {
        assert(isValid)
        assert(!inputs.isEmpty && !outputNames.isEmpty)

        var status: OrtStatusPtr?
        var outputTensors: [OpaquePointer?] = Array(repeating: nil, count: outputNames.count)

        let inputNames = inputs.map { $0.name }
        let inputTensors = inputs.map { $0.ptr }

        withArrayOfCStrings(inputNames) { inNames in
            withArrayOfCStrings(outputNames) { outNames in
                status = self.api.pointee.Run(
                    self.session,
                    nil,
                    inNames,
                    inputTensors,
                    inputNames.count,
                    outNames,
                    outputNames.count,
                    &outputTensors
                )
            }
        }

        if let status {
            let message = String(cString: api.pointee.GetErrorMessage(status)!)
            api.pointee.ReleaseStatus(status)
            throw .Status("Failed to Run \(message)")
        }

        var result: [String: OrtTensor] = [:]
        for (i, name) in outputNames.enumerated() {
            guard let tensor = outputTensors[i] else { continue }

            var info: OpaquePointer?
            status = api.pointee.GetTensorMemoryInfo(tensor, &info)

            if let status {
                let message = String(cString: api.pointee.GetErrorMessage(status)!)
                api.pointee.ReleaseStatus(status)
                throw .Status("Failed to Get MemoryInfo \(message)")
            }

            result[name] = OrtTensor(name: name, ptr: tensor)
        }

        return result
    }

    public func GetData<DataType: Numeric>(
        from tensor: OrtTensor,
        into data: inout [DataType],
        size: Int
    ) throws(OrtError) {
        assert(tensor.isValid)
        var status: OrtStatusPtr?

        var info: OpaquePointer?
        status = api.pointee.GetTensorTypeAndShape(tensor.ptr, &info)
        if let status {
            let message = String(cString: api.pointee.GetErrorMessage(status)!)
            api.pointee.ReleaseStatus(status)
            throw .Status("Failed to get tensor type and shape info \(message)")
        }

        var count = 0
        status = api.pointee.GetTensorShapeElementCount(info, &count)
        if let status {
            let message = String(cString: api.pointee.GetErrorMessage(status)!)
            api.pointee.ReleaseStatus(status)
            throw .Status("Failed to get tensor element count \(message)")
        }
        assert(count == size)

        var pointer: UnsafeMutableRawPointer?
        status = api.pointee.GetTensorMutableData(tensor.ptr, &pointer)

        if let status {
            let message = String(cString: api.pointee.GetErrorMessage(status)!)
            api.pointee.ReleaseStatus(status)
            throw .Status("Failed to get date from tensor \(message)")
        }
        assert(pointer != nil)

        var out: [DataType] = Array(repeating: 0, count: size)
        if let dataPtr = pointer?.assumingMemoryBound(to: DataType.self) {
            out = Array(UnsafeBufferPointer(start: dataPtr, count: size))
        } else {
            throw .Status("Failed to convert data type")
        }

        data = out
    }
}

public func MakeONNX() -> ONNX {
    return ONNX()
}

func withArrayOfCStrings<R>(
    _ args: [String], _ body: ([UnsafePointer<CChar>?]) -> R
) -> R {
    let argsCounts: [Int] = Array(args.map { $0.utf8.count + 1 })
    let argsOffsets = [0] + scan(argsCounts, 0, +)
    let argsBufferSize: Int = argsOffsets.last!

    var argsBuffer: [UInt8] = []
    argsBuffer.reserveCapacity(argsBufferSize)
    for arg: String in args {
        argsBuffer.append(contentsOf: arg.utf8)
        argsBuffer.append(0)
    }

    return argsBuffer.withUnsafeMutableBufferPointer { argsBuffer in
        let ptr: UnsafePointer<CChar> = UnsafeRawPointer(argsBuffer.baseAddress!).bindMemory(
            to: CChar.self, capacity: argsBuffer.count
        )
        var cStrings: [UnsafePointer<CChar>?] = argsOffsets.map { ptr + $0 }
        cStrings.append(nil)
        return body(cStrings)
    }
}

func scan<S: Sequence, U>(
    _ seq: S, _ initial: U, _ combine: (U, S.Iterator.Element) -> U
) -> [U] {
    var result: [U] = []
    result.reserveCapacity(seq.underestimatedCount)
    var runningResult: U = initial
    for element: S.Element in seq {
        runningResult = combine(runningResult, element)
        result.append(runningResult)
    }
    return result
}
