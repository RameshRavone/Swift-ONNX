// tensor.swift
// Copyright (c) 2025 FrogSquare
// Created by Ramesh (Ravone)

import CONNX

@frozen
public struct OrtTensor {
    var name: String
    var ptr: OpaquePointer?

    var isValid: Bool {
        ptr != nil
    }

    func checkStatus(api: UnsafePointer<OrtApi>, _ status: OrtStatusPtr?, _ message: String = "") throws {
        if let status {
            let msg = String(cString: api.pointee.GetErrorMessage(status)!)
            api.pointee.ReleaseStatus(status)
            throw message + "::" + msg
        }
    }

    public func GetData<DataType: Numeric>(
        api: UnsafePointer<OrtApi>,
        into data: inout [DataType],
        size: Int
    ) throws {
        assert(isValid)
        var status: OrtStatusPtr?

        var info: OpaquePointer?
        status = api.pointee.GetTensorTypeAndShape(ptr, &info)
        try checkStatus(api: api, status, "Failed to get tensor type and shape info")

        var count = 0
        status = api.pointee.GetTensorShapeElementCount(info, &count)
        try checkStatus(api: api, status, "Failed to get element count")

        assert(count == size)

        var pointer: UnsafeMutableRawPointer?
        status = api.pointee.GetTensorMutableData(ptr, &pointer)
        try checkStatus(api: api, status, "Failed to get data from tensor")

        assert(pointer != nil)

        var out: [DataType] = Array(repeating: 0, count: size)
        if let dataPtr = pointer?.assumingMemoryBound(to: DataType.self) {
            out = Array(UnsafeBufferPointer(start: dataPtr, count: size))
        } else {
            throw "Failed to convert data type"
        }

        data = out
    }
}
