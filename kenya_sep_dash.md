# Kenya September Dashboard Update Notes

This dashboard update separates the cost-effectiveness logic and DALY logic from the Streamlit plotting code so future dashboard updates can be smaller and easier to review.

## What Was Updated

### Cost

- Added `cost.py` as the central place for SDR cost-effectiveness calculations.
- Moved the corrected Tingting-style cost calculation into functions that use dashboard dataframes directly:
  - `b_df`
  - `i_df`
- Removed the need to generate intermediate CSV files before calculating cost.
- Cost effectiveness is guarded as Kakamega-only in the dashboard.
- `SDR_Dash.py` and `SDR_Dash_TI.py` now import cost functions instead of calculating cost inline.

### DALYs

- Added `daly.py` as the central place for DALY calculations.
- Updated maternal DALYs to use the corrected structure:
  - `M_YLL`: years of life lost from maternal death
  - `M_YLD`: years lived with disability from nonfatal maternal complications
  - `M_DALY = M_YLL + M_YLD`
- Kept existing neonatal DALY logic unchanged.
- Updated `model_run.py` to output:
  - `M_DALYs`
  - `M_YLLs`
  - `M_YLDs`
  - `N_DALYs`
  - `DALYs`
- DALYs apply to all counties.

## Files To Check When Updating The Dashboard

- `cost.py`: update cost assumptions or cost formulas here.
- `daly.py`: update DALY formulas here.
- `model_run.py`: update only if model output columns need to change.
- `SDR_Dash.py` / `SDR_Dash_TI.py`: update only display, plotting, or dashboard flow.

## How To Run

```bash
cd /Users/poppy/Documents/GitHub/sdr-dashboard-deploy
streamlit run SDR_Dash_TI.py
```

Or:

```bash
streamlit run SDR_Dash.py
```
