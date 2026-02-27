$source = Get-Location
$zipFile = "archive.zip"
$excludeDirs = @(".git", ".venv")

$items = Get-ChildItem -Path $source -Recurse | Where-Object {
    $excludeDirs -notcontains $_.Parent.Name -and
    $excludeDirs -notcontains $_.Name
}

# Remove existing zip if present
if (Test-Path $zipFile) { Remove-Item $zipFile }

Compress-Archive -Path $items.FullName -DestinationPath $zipFile