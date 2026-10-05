@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64 -no_logo >nul
cmake --build D:\llm\build\qc-mtp --target llama-server llama-bench -j 4
