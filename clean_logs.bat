@echo off
echo Cleaning up TensorBoard logs...

:: /s: Deletes specified files from all subdirectories
:: /q: Quiet mode, won't ask for confirmation for each file
del /s /q events.out.tfevents.*

echo Done. All tfevents files have been purged.
pause