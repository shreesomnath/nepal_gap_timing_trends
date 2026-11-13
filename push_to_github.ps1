param(
    [string]$GitHubUser = "shreesomnath",
    [string]$RepoName = "nepal_gap_timing_trends",
    [string]$AuthorName = "Somnath Luitel",
    [string]$AuthorEmail = "somnath.luitel416@topper.wku.edu"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".git")) { git init -b main | Out-Null }
git config user.name $AuthorName
git config user.email $AuthorEmail

git add -A
# refuse to continue if any restricted file is staged
$blocked = git diff --cached --name-only | Where-Object { $_ -match 'melted_ppt|Precipitation_|\.xlsx$|daily_qc\.npz|qc_flags\.csv|neighbour_screen_pre2018\.csv' }
if ($blocked) { Write-Error "Restricted files staged, stopping: $blocked" }

git commit -m "Code and derived data for 'Missing-Data Timing Alters Station Rainfall Trend Estimates in Nepal' (v1.0.0)"
$remote = "https://github.com/$GitHubUser/$RepoName.git"
if (-not (git remote | Select-String -SimpleMatch "origin")) { git remote add origin $remote }
git push -u origin main
Write-Host "Pushed to $remote. Next: enable the repository in Zenodo and publish release v1.0.0 (see PUSH_TO_GITHUB.md)."
