// lib.swift
// Copyright (c) 2025 FrogSquare
// Created by Ramesh (Ravone)

import CONNX
import Foundation

extension Swift.String: Swift.Error {}

typealias ORT_API = UnsafePointer<OrtApi>?

@frozen
public struct ONNX {
    var api: UnsafePointer<OrtApi>

    var env: OpaquePointer?
    var session: OpaquePointer?
    var sessionOptions: OpaquePointer?

    var inputCount: Int = 0

    public var isValid: Bool { env != nil && session != nil }

    init() {
        api = OrtGetApiBase()!.pointee.GetApi(UInt32(ORT_API_VERSION))!
    }

    func checkStatus(_ status: OrtStatusPtr?, _ message: String = "") throws {
        if let status {
            let msg = String(cString: api.pointee.GetErrorMessage(status)!)
            api.pointee.ReleaseStatus(status)
            throw message + "::" + msg
        }
    }

    public mutating func Release(tensor: OrtTensor) {
        assert(tensor.isValid)
        api.pointee.ReleaseValue(tensor.ptr)
    }

    public mutating func Release() {
        if let sessionOptions {
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

    public mutating func CreateEnv(name: String = "ONNX_BASE") throws {
        let status = api.pointee.CreateEnv(ORT_LOGGING_LEVEL_ERROR, name, &env)
        try checkStatus(status, "Cannot create Environment")

        assert(env != nil)
    }

    public mutating func CreateSession(model path: String, usingGPU gpu: Bool = false) throws {
        assert(env != nil, "Call CreateEnv() before Creating a Session")

        var status: OrtStatusPtr?
        var sessionOptions: OpaquePointer?
        if gpu {
            status = api.pointee.CreateSessionOptions(&sessionOptions)
            try checkStatus(status, "Cannot create Session")

            try checkStatus(api.pointee.SetIntraOpNumThreads(sessionOptions, 1))
            try checkStatus(api.pointee.SetSessionGraphOptimizationLevel(sessionOptions, ORT_DISABLE_ALL))
            // try checkStatus(api.pointee.DisableMemPattern(sessionOptions))
            // try checkStatus(api.pointee.DisableCpuMemArena(sessionOptions))

            var o = OrtCUDAProviderOptions()
            o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault
            o.gpu_mem_limit = 1024 * 1024 * 1024
            o.tunable_op_enable = 0
            o.tunable_op_tuning_enable = 0

            status = api.pointee.SessionOptionsAppendExecutionProvider_CUDA(
                sessionOptions,
                &o
            )
            try checkStatus(status, "Cannot Enable CUDA")
        }

        status = api.pointee.CreateSession(env, path, sessionOptions, &session)
        if let status {
            let message = String(cString: api.pointee.GetErrorMessage(status)!)

            api.pointee.ReleaseEnv(env)
            api.pointee.ReleaseStatus(status)
            throw "Cannot load model: \(message)"
        }
        assert(session != nil)

        try checkStatus(api.pointee.SessionGetInputCount(session, &inputCount))
    }

    public func CreateInput(
        name: String,
        data: UnsafeMutableRawPointer,
        shape: [Int64]
    ) throws -> OrtTensor {
        assert(env != nil && session != nil)

        var status: OrtStatusPtr?
        var memoryInfo: OpaquePointer?
        var inputTensor: OpaquePointer?

        /*
         do {
             var allocator: UnsafeMutablePointer<OrtAllocator>?
             status = api.pointee.GetAllocatorWithDefaultOptions(&allocator)
             try checkStatus(status)

             var inputNamePtr: UnsafeMutablePointer<Int8>?
             try checkStatus(api.pointee.SessionGetInputName(session, 0, allocator, &inputNamePtr))
             let inputName = String(cString: inputNamePtr!)
             print("INputName: \(inputName)")
             allocator!.pointee.Free.unsafelyUnwrapped(allocator, inputNamePtr)
         }
         */

        status = api.pointee.CreateCpuMemoryInfo(
            OrtArenaAllocator,
            OrtMemTypeDefault,
            &memoryInfo
        )
        try checkStatus(status, "Cannot Create memoryInfo")

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
        try checkStatus(status, "Cannot Create Input Tensor")

        api.pointee.ReleaseMemoryInfo(memoryInfo)
        return OrtTensor(name: name, ptr: inputTensor)
    }

    public func Run(
        withInputs inputs: [OrtTensor],
        outputNames: [String]
    ) throws -> [String: OrtTensor] {
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
        try checkStatus(status, "Failed to RUN")

        var result: [String: OrtTensor] = [:]
        for (i, name) in outputNames.enumerated() {
            guard let tensor = outputTensors[i] else { continue }

            result[name] = OrtTensor(name: name, ptr: tensor)
        }

        return result
    }

    public func GetData<DataType: Numeric>(
        from tensor: OrtTensor,
        into data: inout [DataType],
        size: Int
    ) throws {
        assert(tensor.isValid)
        try tensor.GetData(api: api, into: &data, size: size)
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
