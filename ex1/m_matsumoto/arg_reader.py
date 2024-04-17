"""
Arguments Reader

This module reads 
- input path
- output path
from argments. 
"""

import argparse

"""
Reads IO-path
"""
def io_path(isHasInput:bool, isHasOutput:bool):
    # パーサー初期化処理
    parser = argparse.ArgumentParser(
        description="Reads IO file path"
    )

    # 入力パスあり
    if(isHasInput):
        parser.add_argument("--input-path", type=str, required=True, help="input path")
    
    # 出力パスあり        
    if(isHasOutput):
        parser.add_argument("--output-path", type=str, required=True, help="output path")

    return parser.parse_args()



if __name__ == "__main__":
    args = io_path(True, False)
    print(args.input_path)