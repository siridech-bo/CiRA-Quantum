# Real-world forecasting datasets (QRC benchmark)

Used by `backend/scripts/qrc_realworld_bench.py`. Downloaded 2026-10-22.
Raw files live in `raw/` (not committed where the license restricts
redistribution; fetch them with the commands below).

| Dataset | Source | License | Committed |
|---|---|---|---|
| ERA5 daily T + RH, Delhi / Mumbai / Bangkok / Beijing / Berlin, 1970-2025 | Open-Meteo Historical Weather API (`models=era5`) | CC BY 4.0 (Open-Meteo / Copernicus ERA5) | yes, `../weather/era5/*.json` |
| Santa Fe laser A (A.dat 1,000 + A.cont 9,093 points) | Santa Fe Time Series Competition (Weigend & Gershenfeld 1993), archived Stanford mirror | research benchmark, no formal license | yes, `raw/santafe_A.*` |
| Beijing PM2.5 (hourly, 2010-2014; TEMP, DEWP used) | UCI ML Repository #381 (Liang et al. 2015) | CC BY 4.0 | no (fetch) |
| ETTh1 (hourly, 2016-07 .. 2018-06; OT, HUFL used) | ETDataset, Zhou et al. (Informer, AAAI 2021) | CC BY-ND 4.0 | no (fetch) |

ERA5 request (one call per city, change lat/lon/timezone):

    https://archive-api.open-meteo.com/v1/archive?latitude=28.61&longitude=77.21&start_date=1970-01-01&end_date=2025-12-31&daily=temperature_2m_mean,relative_humidity_2m_mean,dew_point_2m_mean&models=era5&timezone=Asia%2FKolkata

Cities: Delhi 28.61,77.21 (Asia/Kolkata); Mumbai 19.08,72.88 (Asia/Kolkata);
Bangkok 13.75,100.50 (Asia/Bangkok); Beijing 39.90,116.40 (Asia/Shanghai);
Berlin 52.52,13.40 (Europe/Berlin). The free API allows ~1 request/minute burst.

Other raw files:

    curl -L https://web.archive.org/web/2016/http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe/A.dat  -o raw/santafe_A.dat
    curl -L https://web.archive.org/web/2016/http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe/A.cont -o raw/santafe_A.cont
    curl -L https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip -o raw/beijing_pm25.zip   # unzip in raw/
    curl -L https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv -o raw/ETTh1.csv

Caveat: ERA5 is reanalysis on a ~0.25° grid (smoother than station data);
report it as "ERA5 <city>", separately from the Kaggle station task.
