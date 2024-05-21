
from project_imports import *


# Define the function to calculate the labels that we wish to predict
def calculate_returns(stock):
    stock['ret5'] = stock['prc'].pct_change(5, fill_method=None).shift(-5) #* 100
    stock['ret20'] = stock['prc'].pct_change(20, fill_method=None).shift(-20) #* 100
    stock['ret60'] = stock['prc'].pct_change(60, fill_method=None).shift(-60) #* 100
    return stock


# Define the function to load the sp500 data
def load_sp500_data(conn, start_date, end_date, freq='daily'):

    ##########################################
    # Get a list of                          #
    # S&P 500 Index Constituents             #
    ##########################################

    sql_engine = conn.engine
    connection = sql_engine.raw_connection()

    if freq == 'daily':

        # Query using raw_sql method
        query = f"""
            SELECT a.*, b.date, b.prc, b.openprc, b.ret, 
                b.askhi, b.bidlo, b.vol, b.shrout, b.cfacpr, b.cfacshr
            FROM crsp.msp500list AS a,
                crsp.dsf AS b
            WHERE a.permno = b.permno
            AND b.date >= a.start
            AND b.date <= a.ending
            AND b.date >= '{start_date}'
            AND b.date <= '{end_date}'
            ORDER BY date;
        """

        sp500 = pd.read_sql_query(query, connection, parse_dates=['start', 'ending', 'date'])

        # sp500 = conn.raw_sql(f"""
        #                         SELECT a.*, b.date, b.prc, b.openprc, b.ret, 
        #                         b.askhi, b.bidlo, b.vol, b.shrout, b.cfacpr, b.cfacshr
        #                         FROM crsp.msp500list AS a,
        #                         crsp.dsf AS b
        #                         WHERE a.permno = b.permno
        #                         AND b.date >= a.start AND b.date <= a.ending
        #                         AND b.date >= '{start_date}'
        #                         AND b.date <= '{end_date}'
        #                         ORDER BY date;
        #                         """, date_cols=['start', 'ending', 'date'])
        
        # Calculate the adjusted prices
        sp500[['prc', 'openprc', 'askhi', 'bidlo']] = sp500[['prc', 'openprc', 'askhi', 'bidlo']].div(sp500['cfacpr'], axis=0)
        
    elif freq == 'monthly':

        # Query using raw_sql method
        query = f"""
            SELECT a.*, b.date, b.prc, b.ret, 
                b.askhi, b.bidlo, b.vol, b.shrout, b.cfacpr, b.cfacshr
            FROM crsp.msp500list AS a,
                crsp.msf AS b
            WHERE a.permno = b.permno
            AND b.date >= a.start
            AND b.date <= a.ending
            AND b.date >= '{start_date}'
            AND b.date <= '{end_date}'
            ORDER BY date;
        """

        sp500 = pd.read_sql_query(query, connection, parse_dates=['start', 'ending', 'date'])

        # sp500 = conn.raw_sql(f"""
        #                         SELECT a.*, b.date, b.prc, b.ret, 
        #                         b.askhi, b.bidlo, b.vol, b.shrout, b.cfacpr, b.cfacshr
        #                         FROM crsp.msp500list AS a,
        #                         crsp.msf AS b
        #                         WHERE a.permno = b.permno
        #                         AND b.date >= a.start AND b.date <= a.ending
        #                         AND b.date >= '{start_date}'
        #                         AND b.date <= '{end_date}'
        #                         ORDER BY date;
        #                         """, date_cols=['start', 'ending', 'date'])

        # Calculate the adjusted prices
        sp500[['prc', 'askhi', 'bidlo']] = sp500[['prc', 'askhi', 'bidlo']].div(sp500['cfacpr'], axis=0)

    # Calculate the adjusted volume
    sp500['vol'] = sp500['vol'] * sp500['cfacshr']

    # Add Other Descriptive Variables

    # Query using raw_sql method
    query = f"""
            SELECT comnam, ncusip, namedt, nameendt, 
            permno, shrcd, exchcd, hsiccd, ticker
            FROM crsp.msenames
            WHERE permno IN ({",".join([str(permno) for permno in sp500.permno.unique()])})
            """
    
    mse = pd.read_sql_query(query, connection, parse_dates=['namedt', 'nameendt'])

    # mse = conn.raw_sql(f"""
    #                     SELECT comnam, ncusip, namedt, nameendt, 
    #                     permno, shrcd, exchcd, hsiccd, ticker
    #                     FROM crsp.msenames
    #                     WHERE permno IN ({",".join([str(permno) for permno in sp500.permno.unique()])})
    #                     """, date_cols=['namedt', 'nameendt'])

    # if nameendt is missing then set to today date
    mse['nameendt'] = mse['nameendt'].fillna(pd.to_datetime('today'))

    # Merge with SP500 data
    sp500_full = pd.merge(sp500, mse, how = 'left', on = 'permno')

    # Impose the date range restrictions
    sp500_full = sp500_full.loc[(sp500_full.date>=sp500_full.namedt)
                                & (sp500_full.date<=sp500_full.nameendt)]
    

    # Calculate the returns as the labels to predict
    sp500_full_ret = sp500_full.sort_values('date').groupby('permno').apply(calculate_returns)
    
    # Reset the indexes and drop the old index
    sp500_full_ret = sp500_full_ret.reset_index(drop=True)

    return sp500_full_ret


# Define the function to load the data
def load_ff5_data(conn, start_date, end_date):

    sql_engine = conn.engine
    connection = sql_engine.raw_connection()

    # Query using raw_sql method
    query = f"""
            select *
            from ff.fivefactors_daily
            where date>='{start_date}'
            AND date <= '{end_date}'
            order by date;
            """

    ff5 = pd.read_sql_query(query, connection, parse_dates=['date'])

    # ff5 = conn.raw_sql(f"""
    #                 select *
    #                 from ff.fivefactors_daily
    #                 where date>='{start_date}'
    #                 AND date <= '{end_date}'
    #                 order by date;
    #                 """, date_cols=['date'])
    
    return ff5



# Define the function to load the firm characteristics data
def load_firm_char_data(start_date, end_date, permno_unique=[]):

    # Path to the ZIP file
    zip_path = f"{DATA_PATH}/firm_characteristics/PredictorsIndiv.zip"

    # Temporary directory to extract files
    with TemporaryDirectory() as temp_dir:
        # Open the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all files to the temporary directory
            zip_ref.extractall(temp_dir)
            
            # Get a list of CSV files in the directory (assuming all files are CSVs)
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
            
            # Ensure we only process the first 5 CSV files
            # csv_files = csv_files[:5]
            
            # Initialize an empty DataFrame for the combined data
            combined_df = None
            
            # Iterate over the CSV files
            for csv_file in tqdm(csv_files, desc="Processing CSV files"):
                df = pd.read_csv(os.path.join(temp_dir, csv_file))

                if len(permno_unique) != 0:
                    # Filter the dataframe to include only rows with 'permno' in permno_unique
                    df = df[df['permno'].isin(permno_unique)]
                
                df = df[df['yyyymm'] > start_date]
                df = df[df['yyyymm'] < end_date]

                # If combined_df is not yet initialized, use the first dataframe
                if combined_df is None:
                    combined_df = df
                else:
                    # Merge the current dataframe with the combined dataframe on 'permno' and 'yyyymm'
                    combined_df = pd.merge(combined_df, df, on=['permno', 'yyyymm'], how='outer')
            
            # Optional: Save the combined dataframe to a new CSV file
            combined_df.to_csv(f"{DATA_PATH}/firm_characteristics/combined_df.csv", index=False)
            
            print("Combined CSV file created successfully.")

    return combined_df
















        