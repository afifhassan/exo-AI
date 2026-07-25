import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

def fetch_nasa_table_csv(table_name, description):
    print(f"📡 Requesting {description} ({table_name}) via Fast TAP CSV API...")
    
    # Encode query to URL format
    sql_query = f"select * from {table_name}"
    encoded_query = urllib.parse.quote(sql_query)
    
    # NASA Direct TAP API endpoint requesting CSV output
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={encoded_query}&format=csv"
    
    # Load directly into Pandas
    df = pd.read_csv(url)
    print(f"✅ Successfully downloaded {len(df)} rows and {len(df.columns)} columns.\n")
    return df

if __name__ == "__main__":
    print("🚀 Initializing NASA Fast Data Pipeline...\n")
    
    # 1. Fetch satellite tables as CSV
    df_master = fetch_nasa_table_csv("ps", "Confirmed Planetary Systems")
    df_kepler = fetch_nasa_table_csv("cumulative", "Kepler Candidates")
    df_tess = fetch_nasa_table_csv("toi", "TESS Candidates")
    df_k2 = fetch_nasa_table_csv("k2pandc", "K2 Candidates")

    # 2. Save to local SQLite database
    print("💾 Writing data to SQLite Database ('exoplanet_ai_core.db')...")
    engine = create_engine('sqlite:///exoplanet_ai_core.db')
    
    df_master.to_sql('planetary_systems', engine, if_exists='replace', index=False)
    df_kepler.to_sql('kepler_candidates', engine, if_exists='replace', index=False)
    df_tess.to_sql('tess_candidates', engine, if_exists='replace', index=False)
    df_k2.to_sql('k2_candidates', engine, if_exists='replace', index=False)
    
    print("🎉 Pipeline Complete! Your database is ready.")