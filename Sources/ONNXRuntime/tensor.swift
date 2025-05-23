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

    public func GetData<DataType: Numeric>(
        api: UnsafePointer<OrtApi>,
        into data: inout [DataType],
        size: Int
    ) throws(OrtError) {
        assert(isValid)
        var status: OrtStatusPtr?

        var info: OpaquePointer?
        status = api.pointee.GetTensorTypeAndShape(ptr, &info)
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
        status = api.pointee.GetTensorMutableData(ptr, &pointer)

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
