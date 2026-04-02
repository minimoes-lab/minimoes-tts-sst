#!/usr/bin/env python3
"""
Patch script for qwen_tts check_model_inputs decorator bug.
Run this after pip install to fix the broken decorator in rekuenkdr fork.
"""
import os
import sys

def patch_qwen_tts_decorator():
    """
    Fix the @check_model_inputs() decorator bug by patching the source file.
    Changes @check_model_inputs() to @check_model_inputs (removes parentheses).
    """
    try:
        import qwen_tts
        qwen_tts_path = os.path.dirname(qwen_tts.__file__)
        target_file = os.path.join(
            qwen_tts_path, 
            'core', 'tokenizer_12hz', 
            'modeling_qwen3_tts_tokenizer_v2.py'
        )
        
        if not os.path.exists(target_file):
            print(f"[PATCH] Target file not found: {target_file}")
            return False
        
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_count = content.count('@check_model_inputs()')
        
        if original_count == 0:
            print(f"[PATCH] No @check_model_inputs() found - already patched or different version")
            return True
        
        patched = content.replace('@check_model_inputs()', '@check_model_inputs')
        
        old_def = '''def check_model_inputs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper'''
        
        new_def = '''def check_model_inputs(func=None, **kwargs):
    """PEP 318 compliant - works with @decorator and @decorator()."""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs_inner):
            return f(*args, **kwargs_inner)
        return wrapper
    if func is not None:
        return decorator(func)
    return decorator'''
        
        if old_def in patched:
            patched = patched.replace(old_def, new_def)
            print(f"[PATCH] Fixed decorator definition")
        
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(patched)
        
        print(f"[PATCH] Fixed {original_count} decorator(s) in {target_file}")
        return True
        
    except Exception as e:
        print(f"[PATCH] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = patch_qwen_tts_decorator()
    sys.exit(0 if success else 1)
