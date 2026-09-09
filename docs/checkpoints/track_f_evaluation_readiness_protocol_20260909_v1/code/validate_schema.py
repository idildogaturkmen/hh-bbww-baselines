#!/usr/bin/env python3
"""Minimal, dependency-free schema validator (this environment has no
`jsonschema` package installed -- confirmed this session, not assumed).
Does NOT implement full JSON Schema draft 2020-12 semantics -- checks only
what this package's own schemas actually need: top-level `required` keys
present, and, where a schema gives a `type`, a basic type match (including
walking `properties`/`items` one level at a time, recursively). This is
intentionally a real, working, conservative subset, not a stub -- it will
correctly catch a missing required field or an obviously wrong type
(string where a number is required, etc.), which is what this package's
own tests exercise it against.

Usage:
    python3 validate_schema.py --schema schemas/comparison_result.schema.json --instance result.json
Exit 0 = valid (per this validator's checks), 1 = invalid (errors printed), 2 = usage/IO error.
"""
import argparse
import json
import sys

_TYPE_MAP = {
    "object": dict, "array": list, "string": str,
    "number": (int, float), "integer": int, "boolean": bool, "null": type(None),
}


def _check_type(value, type_spec, path, errors):
    if type_spec is None:
        return
    types = type_spec if isinstance(type_spec, list) else [type_spec]
    if not any(isinstance(value, _TYPE_MAP[t]) for t in types if t in _TYPE_MAP):
        errors.append(f"{path}: expected type {types}, got {type(value).__name__}")


def _resolve(schema, root):
    if isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        assert ref.startswith("#/"), f"only local $ref supported, got {ref}"
        node = root
        for part in ref[2:].split("/"):
            node = node[part]
        return node
    return schema


def validate(instance, schema, root=None, path="$", errors=None):
    if errors is None:
        errors = []
    if root is None:
        root = schema
    schema = _resolve(schema, root)

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {instance!r}")

    if "type" in schema:
        _check_type(instance, schema["type"], path, errors)

    if isinstance(instance, dict):
        for req in schema.get("required", []):
            if req not in instance:
                errors.append(f"{path}: missing required property '{req}'")
        props = schema.get("properties", {})
        for key, value in instance.items():
            if key in props:
                validate(value, props[key], root, f"{path}.{key}", errors)

    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            validate(item, schema["items"], root, f"{path}[{i}]", errors)
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: expected at least {schema['minItems']} items, got {len(instance)}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: expected at most {schema['maxItems']} items, got {len(instance)}")

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in enum {schema['enum']}")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--instance", required=True)
    args = ap.parse_args()

    with open(args.schema) as f:
        schema = json.load(f)
    with open(args.instance) as f:
        instance = json.load(f)

    errors = validate(instance, schema)
    if errors:
        print(f"INVALID: {args.instance} against {args.schema} -- {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    print(f"VALID: {args.instance} conforms to {args.schema} (per this package's lightweight validator)")
    sys.exit(0)


if __name__ == "__main__":
    main()
