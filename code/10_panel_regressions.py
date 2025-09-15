
import argparse, pandas as pd, numpy as np
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
df=pd.read_csv(a.data)
df['quarter']=pd.PeriodIndex(df['quarter'],freq='Q')
df=df.sort_values(['firm_id','quarter']).set_index(['firm_id','quarter'])
for c in ['assets_bil','roa','leverage','rev_growth']:
    if c not in df.columns: df[c]=0.0
    df[c]=df[c].fillna(0)
q25,q75=df['sc_intensity'].quantile(0.25),df['sc_intensity'].quantile(0.75)
df['sc_norm']=(df['sc_intensity']-q25)/(q75-q25+1e-9)
df['g_sc']=df['gscpi'].fillna(0)*df['sc_norm'].fillna(0)
y=df['ccc_days'] if 'ccc_days' in df.columns else (df['dio']+df['dso']-df['dpo'])
X=sm.add_constant(df[['gscpi','g_sc','assets_bil','roa','leverage','rev_growth']].fillna(0))
res=PanelOLS(y,X,entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
Path(a.out).mkdir(parents=True, exist_ok=True)
open(Path(a.out)/'table4_col3.txt','w').write(str(res.summary))
print(res.summary)
