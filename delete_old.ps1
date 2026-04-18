$targetDir = "results"
# PowerShell will parse this based on the machine's local time zone. 
# If your machine is in a different time zone than CST, adjust the time accordingly.
$targetDate = [datetime]"2026-04-17 12:00:00"

Write-Host "Deleting files in $targetDir older than $targetDate..."

# -Recurse: searches all subfolders
# -File: ignores directories themselves, only targets files
# $_.LastWriteTime -lt $targetDate: filters for files older than the target
# Remove-Item -Force: deletes the files, bypassing read-only flags
Get-ChildItem -Path $targetDir -Recurse -File | 
    Where-Object { $_.LastWriteTime -lt $targetDate } | 
    Remove-Item -Force

Write-Host "Cleanup complete."

# Optional: To remove leftover empty directories, uncomment the lines below:
Get-ChildItem -Path $targetDir -Recurse -Directory | 
    Where-Object { (Get-ChildItem -Path $_.FullName).Count -eq 0 } | 
    Remove-Item -Force