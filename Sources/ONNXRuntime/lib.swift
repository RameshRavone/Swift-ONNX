// lib.swift
// Copyright (c) 2025 FrogSquare
// Created by Ramesh (Ravone)

import CONNX
import Foundation

typealias ORT_API = UnsafePointer<OrtApi>?

public enum OrtError: Error {
    case Status(String)
    case OutputParsingError
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

    public func Release(value: OpaquePointer?) {
        assert(value != nil, "Value is a nil")
        api.pointee.ReleaseValue(value)
    }

    public func Release() {
        if usingGPU, let sessionOptions {
            api.pointee.ReleaseSessionOptions(sessionOptions)
        }
        if let session {
            api.pointee.ReleaseSession(session)
        }
        if let env {
            api.pointee.ReleaseEnv(env)
        }
    }

    public mutating func CreateEnv(name: String = "ONNX_BASE") throws(OrtError) {
        if let status: OrtStatusPtr = api.pointee.CreateEnv(ORT_LOGGING_LEVEL_WARNING, name, &env) {
            api.pointee.ReleaseStatus(status)
            let message = String(cString: api.pointee.GetErrorMessage(status)!)
            throw .Status("Cannot create Environment \(message)")
        }
        assert(env != nil)
    }

    public mutating func CreateSession(model path: String, usingGPU gpu: Bool = false) throws(OrtError) {
        assert(env != nil, "Call CreateEnv() before Creating a Session")

        usingGPU = gpu

        var sessionOptions: OpaquePointer?
        if usingGPU {
            _ = api.pointee.CreateSessionOptions(&sessionOptions)

            var o = OrtCUDAProviderOptions()
            o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive
            o.gpu_mem_limit = Int(SIZE_MAX)

            if let cudaStatus: OrtStatusPtr = api.pointee.SessionOptionsAppendExecutionProvider_CUDA(
                sessionOptions,
                &o
            ) {
                let message = String(cString: api.pointee.GetErrorMessage(cudaStatus)!)
                api.pointee.ReleaseStatus(cudaStatus)
                throw .Status("Cannot Enable CUDA \(message)")
            }
        }

        if let status: OrtStatusPtr = api.pointee.CreateSession(env, path, sessionOptions, &session) {
            let message = String(cString: api.pointee.GetErrorMessage(status)!)

            api.pointee.ReleaseEnv(env)
            api.pointee.ReleaseStatus(status)
            throw .Status("Cannot load model: \(message)")
        }
        assert(session != nil)
    }

    public func CreateInput(data: [Float32], shape: [Int64]) throws(OrtError) -> OpaquePointer? {
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

        var data = data
        status = api.pointee.CreateTensorWithDataAsOrtValue(
            memoryInfo,
            &data,
            data.count * MemoryLayout<Float32>.size,
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

        return inputTensor
    }

    public func Run(
        withInputs inputs: [String: OpaquePointer?],
        outputNames: [String]?,
        outputCount: Int
    ) throws(OrtError) -> OpaquePointer? {
        assert(!inputs.isEmpty)

        var status: OrtStatusPtr?
        var outputTensor: OpaquePointer?

        let inputNames = Array(inputs.keys)
        let inputTensors = Array(inputs.values)
        withArrayOfCStrings(inputNames) { inputNames in
            withArrayOfCStrings(outputNames ?? ["output"]) { outNames in
                status = self.api.pointee.Run(
                    self.session,
                    nil,
                    inputNames,
                    inputTensors,
                    inputs.count,
                    outNames,
                    outputNames?.count ?? 1,
                    &outputTensor
                )
            }
        }

        if let status {
            let message = String(cString: api.pointee.GetErrorMessage(status)!)
            api.pointee.ReleaseStatus(status)
            throw .Status("Failed to Run \(message)")
        }

        return outputTensor
    }

    public func GetData(
        from tensor: OpaquePointer?,
        into pointer: inout UnsafeMutableRawPointer?
    ) throws(OrtError) {
        assert(tensor != nil)
        if let status = api.pointee.GetTensorMutableData(tensor, &pointer) {
            let message = String(cString: api.pointee.GetErrorMessage(status)!)
            api.pointee.ReleaseStatus(status)
            throw .Status("Failed to get date from tensor \(message)")
        }
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
        cStrings[cStrings.count - 1] = nil
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
