# Publishing this repository (GitHub, then Zenodo)

Nothing here has been pushed. Do these steps yourself when you decide to make the code public.

## 1. Create an empty repository on GitHub
On github.com, create a new repository named `nepal_gap_timing_trends`. Do not add a README, license or .gitignore there; they are already in this folder. Choose **Private** if you want to keep it closed until the paper is accepted, but note that the Zenodo link in step 3 only works for **public** repositories.

## 2. Push from this folder
Run `push_to_github.ps1` in PowerShell from this folder, giving your GitHub user name:

```
powershell -ExecutionPolicy Bypass -File .\push_to_github.ps1 -GitHubUser shreesomnath
```

The script sets your name and e-mail for this repository only, checks that no restricted DHM file is staged, makes one commit and pushes the `main` branch. GitHub will ask you to sign in in the browser the first time.

## 3. Get a DOI from Zenodo
1. Sign in at https://zenodo.org with your GitHub account and open **Settings > GitHub**.
2. Switch the toggle on for `nepal_gap_timing_trends` (the repository must be public).
3. On GitHub, create a release: **Releases > Draft a new release**, tag `v1.0.0`, title `v1.0.0`, then **Publish release**.
4. Zenodo archives the release within a few minutes and shows a DOI. `.zenodo.json` fills the title, authors and license.

## 4. Put the DOI in the paper
Send the DOI back and the Open Research section and reference list will be updated, for example:

> Luitel, S., Biswakarma, K., Thapa, S., Amgain, A., Rayamajhi, D., Bhattarai, M., & Pokharel, Y. (2026). *Missing-data timing and station rainfall trends in Nepal: code and derived data* (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

## Before you push
- Ask all co-authors whether they agree to the public release and the licenses (MIT for code, CC BY 4.0 for derived data).
- Confirm that DHM's terms allow sharing derived annual indices and gap masks. The daily values and monthly totals are excluded and blocked by `.gitignore`.
