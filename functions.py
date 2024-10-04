
from project_imports import *


# Define the function to calculate the labels that we wish to predict
def calculate_returns(stock):
    # stock['ret5'] = stock['prc'].pct_change(5, fill_method=None).shift(-5) #* 100
    stock['ret20'] = stock['prc'].pct_change(20, fill_method=None).shift(-20) #* 100
    # stock['ret60'] = stock['prc'].pct_change(60, fill_method=None).shift(-60) #* 100
    return stock


# Define the function to load the sp500 data
def load_sp500_data(conn, start_date, end_date, freq='daily', add_desc=False):

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

        # Calculate the adjusted prices
        sp500[['prc', 'askhi', 'bidlo']] = sp500[['prc', 'askhi', 'bidlo']].div(sp500['cfacpr'], axis=0)
        
    # Calculate the adjusted volume
    sp500['vol'] = sp500['vol'] * sp500['cfacshr']

    if add_desc:

        # Add Other Descriptive Variables
    
        # Query using raw_sql method
        query = f"""
                SELECT comnam, ncusip, namedt, nameendt, 
                permno, shrcd, exchcd, hsiccd, ticker
                FROM crsp.msenames
                WHERE permno IN ({",".join([str(permno) for permno in sp500.permno.unique()])})
                """
        
        mse = pd.read_sql_query(query, connection, parse_dates=['namedt', 'nameendt'])
    
        # if nameendt is missing then set to today date
        mse['nameendt'] = mse['nameendt'].fillna(pd.to_datetime('today'))
    
        # Merge with SP500 data
        sp500_full = pd.merge(sp500, mse, how = 'left', on = 'permno')
    
        # Impose the date range restrictions
        sp500_full = sp500_full.loc[(sp500_full.date>=sp500_full.namedt)
                                    & (sp500_full.date<=sp500_full.nameendt)]
        
        
        ### USE MONTHLY RETURN DATA
        # # Calculate the returns as the labels to predict
        sp500_full_ret = sp500_full.sort_values('date').groupby('permno').apply(calculate_returns)
        
        # Reset the indexes and drop the old index
        sp500_full_ret = sp500_full_ret.reset_index(drop=True)
    
        return sp500_full_ret
    
    else:
        
        ### USE MONTHLY RETURN DATA
        # # Calculate the returns as the labels to predict
        sp500_ret = sp500.sort_values('date').groupby('permno').apply(calculate_returns)
        
        # Reset the indexes and drop the old index
        sp500_ret = sp500_ret.reset_index(drop=True)
        
        return sp500_ret
    

# Define the function to load the sp500 data
def load_US_market(conn, start_date, end_date, freq='daily', add_desc=False, ret = False):

    ##########################################
    # Get a list of                          #
    # S&P 500 Index Constituents             #
    ##########################################

    sql_engine = conn.engine
    connection = sql_engine.raw_connection()

    if freq == 'daily':

        # Query using raw_sql method
        
        # query = f"""
        #     SELECT a.*, b.date, b.prc, b.openprc, b.ret, 
        #         b.askhi, b.bidlo, b.vol, b.shrout, b.cfacpr, b.cfacshr
        #     FROM crsp.msp500list AS a,
        #         crsp.dsf AS b
        #     WHERE a.permno = b.permno
        #     AND b.date >= a.start
        #     AND b.date <= a.ending
        #     AND b.date >= '{start_date}'
        #     AND b.date <= '{end_date}'
        #     ORDER BY date;
        # """
        
        query = f"""
            SELECT 
                a.permno, 
                a.date,
                a.prc,
                a.openprc,
                a.ret,
                a.askhi,
                a.bidlo,
                a.vol,
                a.shrout, 
                a.cfacpr,
                a.cfacshr,
                b.exchcd
            FROM 
                crsp.dsf AS a
            LEFT JOIN 
                crsp.msenames AS b
            ON 
                a.permno = b.permno 
                AND b.namedt <= a.date 
                AND a.date <= b.nameendt
            WHERE 1=1
                AND a.date >= '{start_date}'
                AND a.date <= '{end_date}'
                AND b.exchcd IN (1, 2, 3)  -- Only select NYSE, NASDAQ, AMEX
                AND b.shrcd IN (10, 11) -- Only select common stock
            ORDER BY 
                a.date;
        """


        US_market = pd.read_sql_query(query, connection, parse_dates=['start', 'ending', 'date'])
        
        # Calculate the adjusted prices
        US_market[['prc', 'openprc', 'askhi', 'bidlo']] = US_market[['prc', 'openprc', 'askhi', 'bidlo']].div(US_market['cfacpr'], axis=0)
        
    elif freq == 'monthly':
        
        query = f"""
            SELECT 
                b.permno, b.date, b.prc, b.ret, b.askhi, b.bidlo, 
                b.vol, b.shrout, b.cfacpr, b.cfacshr, c.exchcd
            FROM 
                crsp.msf AS b
            LEFT JOIN 
                crsp.msenames AS c 
            ON 
                b.permno = c.permno 
                AND c.namedt <= b.date 
                AND b.date <= c.nameendt
            WHERE 1=1
                AND b.date >= '{start_date}'
                AND b.date <= '{end_date}'
                AND c.exchcd IN (1, 2, 3)  -- Only select NYSE, NASDAQ, AMEX
                AND b.shrcd IN (10, 11) -- Only select common stock
            ORDER BY 
                b.date;
        """

        US_market = pd.read_sql_query(query, connection, parse_dates=['start', 'ending', 'date'])

        # Calculate the adjusted prices
        US_market[['prc', 'askhi', 'bidlo']] = US_market[['prc', 'askhi', 'bidlo']].div(US_market['cfacpr'], axis=0)
        
    # Calculate the adjusted volume
    US_market['vol'] = US_market['vol'] * US_market['cfacshr']

    if add_desc:

        # Add Other Descriptive Variables
    
        # Query using raw_sql method
        query = f"""
                SELECT comnam, ncusip, namedt, nameendt, 
                permno, shrcd, exchcd, hsiccd, ticker
                FROM crsp.msenames
                WHERE permno IN ({",".join([str(permno) for permno in sp500.permno.unique()])})
                """
        
        mse = pd.read_sql_query(query, connection, parse_dates=['namedt', 'nameendt'])
    
        # if nameendt is missing then set to today date
        mse['nameendt'] = mse['nameendt'].fillna(pd.to_datetime('today'))
    
        # Merge with SP500 data
        sp500_full = pd.merge(sp500, mse, how = 'left', on = 'permno')
    
        # Impose the date range restrictions
        sp500_full = sp500_full.loc[(sp500_full.date>=sp500_full.namedt)
                                    & (sp500_full.date<=sp500_full.nameendt)]
        
        
        if ret:
            ### USE MONTHLY RETURN DATA
            # # Calculate the returns as the labels to predict
            sp500_full_ret = sp500_full.sort_values('date').groupby('permno').apply(calculate_returns)
        
            # Reset the indexes and drop the old index
            sp500_full_ret = sp500_full_ret.reset_index(drop=True)
    
            return sp500_full_ret
        
        else:
            return sp500_full
    
    else:
        
        if ret:
            ### USE MONTHLY RETURN DATA
            # # Calculate the returns as the labels to predict
            US_market_ret = US_market.sort_values('date').groupby('permno').apply(calculate_returns)
            
            # Reset the indexes and drop the old index
            US_market_ret = US_market_ret.reset_index(drop=True)
            
            return US_market_ret
        else:
            return US_market


# Define the function to load the data
def load_ff5_data(conn, start_date, end_date, freq='daily'):

    sql_engine = conn.engine
    connection = sql_engine.raw_connection()
    
    if freq == 'daily':
        
        # Query using raw_sql method
        query = f"""
                select *
                from ff.fivefactors_daily
                where date>='{start_date}'
                AND date <= '{end_date}'
                order by date;
                """
    
        ff5 = pd.read_sql_query(query, connection, parse_dates=['date'])
        
    elif freq == 'monthly':
        
        # Query using raw_sql method
        query = f"""
                select *
                from ff.fivefactors_monthly
                where date>='{start_date}'
                AND date <= '{end_date}'
                order by date;
                """
    
        ff5 = pd.read_sql_query(query, connection, parse_dates=['date'])
    
    # Create 'yyyymm' column for year-month format
    ff5['yyyymm'] = (ff5['date'].dt.year * 100 + ff5['date'].dt.month).astype('int64')
    
    return ff5



# Define the function to load the firm characteristics data
def load_firm_char_data(start_date, end_date, permno_unique=[]):
    
    # Convert the input date strings to yyyymm integer format
    start_date = int(pd.to_datetime(start_date, format="%d/%m/%Y").strftime('%Y%m'))
    end_date = int(pd.to_datetime(end_date, format="%d/%m/%Y").strftime('%Y%m'))

    # Path to the ZIP file
    zip_path = f"{DATA_PATH}/firm_characteristics/PredictorsIndiv.zip"

    # Temporary directory to extract files
    with TemporaryDirectory() as temp_dir:
        # Open the ZIP file
        with ZipFile(zip_path, 'r') as zip_ref:
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
                
                df = df[df['yyyymm'] >= start_date]
                df = df[df['yyyymm'] <= end_date]

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


def load_returns_crspm(conn, start_date, end_date, rf):

    sql_engine = conn.engine
    connection = sql_engine.raw_connection()
    
    query = f"""
            SELECT 
                a.permno, a.date, a.ret, a.shrout, a.prc, a.altprc, 
                b.exchcd, b.ticker, c.dlstcd, c.dlret 
            FROM 
                crsp.msf AS a
            LEFT JOIN 
                crsp.msenames AS b 
            ON 
                a.permno = b.permno 
            AND b.namedt <= a.date 
            AND a.date <= b.nameendt
            LEFT JOIN 
                crsp.msedelist AS c 
            ON 
                a.permno = c.permno 
                AND date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
            WHERE 
                a.date >= '{start_date}'
            AND a.date <= '{end_date}'
            AND b.exchcd IN (1, 2, 3)  -- Only select NYSE, AMEX, NASDAQ
            AND b.shrcd IN (10, 11) -- Only select common stock
            ORDER BY 
                a.date;
            """
    
    crspm = pd.read_sql_query(query, connection, parse_dates=['start', 'ending', 'date'])
    
    # For testing
    # crspm = crspm2.copy()
    
    # Adjust the return based on the presence of `dlret`, `dlstcd`, and `exchcd`
    # Following  Johnson and Zhao (2007), Shumway and Warther (1999), and Bali, Engle and Murray (2016)
    crspm['dlret'] = np.where(crspm['dlret'].notna(), 
                                crspm['dlret'],  # Use `dlret` if available
                                np.where(((crspm['dlstcd'] == 500) | (crspm['dlstcd'].between(520, 584))),
                                         np.where(crspm['exchcd'].isin([1, 2]), 
                                                  -0.30,  # Assign -30% for NYSE/AMEX (exchcd 1, 2)
                                                  -0.55),  # Assign -55% for NASDAQ (exchcd 3)
                                         np.where(crspm['dlstcd'].notna(), 
                                                  -1.0,  # Assign -100% for all other delisting codes
                                                  crspm['dlret'])))  # Don't change, if there is no delist return or delist code


    # Cap negative delisting returns at -1
    crspm['dlret'] = np.where(
        (crspm['dlret'] < -1) & ~crspm['dlret'].isna(), -1, crspm['dlret']
        )

    # Replace any remaining missing dlret with 0
    crspm['dlret'] = crspm['dlret'].fillna(0)

    # Incorporate delisting return into the regular return
    crspm['ret'] = (1 + crspm['ret']) * (1 + crspm['dlret']) - 1

    # If 'ret' is missing and 'dlret' is non-zero, use 'dlret' as the return
    crspm['ret'] = np.where(crspm['ret'].isna() & (crspm['dlret'] != 0), crspm['dlret'], crspm['ret'])

    # # Convert returns to percentages
    # crspm['ret'] = crspm['ret'] * 100
    
    # Calculate market equity (ME), in millions of dollars
    crspm['me'] = (np.abs(crspm['altprc'] * crspm['shrout']))/1000

    # Create 'yyyymm' column for year-month format
    crspm['yyyymm'] = (crspm['date'].dt.year * 100 + crspm['date'].dt.month).astype('int64')
    
    
    #### Process missing returns
    
    crspm.sort_values(by=['permno', 'date'], inplace=True)
    
    # Shifting the columns
    crspm[['date_ahead']] = crspm.groupby('permno')[['date']].shift(-1)
    
    # Calculate the month difference between 'date' and 'date_ahead'
    crspm['month_diff'] = (crspm['date_ahead'].dt.to_period('M') - crspm['date'].dt.to_period('M')).apply(lambda x: x.n if pd.notna(x) else None)
    
    # Create a condition to identify rows where both the current and previous prices exist
    condition = (~crspm['altprc'].isna()) & (~crspm['altprc'].shift(1).isna())
    
    # Apply the condition to calculate `excess_ret_ahead` where `excess_ret_ahead` is NaN
    crspm['ret'] = np.where(
        crspm['ret'].isna(),
        np.where(
            condition,
            (np.abs(crspm['altprc']) / np.abs(crspm['altprc'].shift(1))) - 1,
            np.nan
        ),
        crspm['ret']
    )
    
    # Drop observations with missing market equity
    crspm = crspm[~crspm.me.isna()]
    
    # # For testing
    # # Step 2: Identify rows with missing `ret` values
    # nan_mask = crspm['ret_ahead'].isna()

    # # Step 3: Create a boolean mask for the row before, the NaN row itself, and the row after
    # nan_or_adjacent = nan_mask | nan_mask.shift(1) | nan_mask.shift(-1) | nan_mask.shift(2) | nan_mask.shift(-2)

    # # Step 4: Filter `crspm` to include only rows matching the mask
    # empty = crspm[nan_or_adjacent]
    
    ####
    
    # Add the risk-free rate to our dataset
    crspm = crspm.merge(rf, on='yyyymm', how='inner')
    
    # Extract the excess return
    crspm['excess_ret'] = crspm['ret'] - crspm['rf']
    
    # Cap negative excess returns at -1
    crspm['excess_ret'] = np.where(
        (crspm['excess_ret'] < -1) & ~crspm['excess_ret'].isna(),  # Cap only if value < -1 and not NaN
        -1, 
        crspm['excess_ret']
    )

    # Shifting the columns
    crspm[['date_ahead', 'excess_ret_ahead']] = crspm.groupby('permno')[['date', 'excess_ret']].shift(-1)
    
    # Re-calculate the month difference between 'date' and 'date_ahead'
    crspm['month_diff'] = (crspm['date_ahead'].dt.to_period('M') - crspm['date'].dt.to_period('M')).apply(lambda x: x.n if pd.notna(x) else None)
    
    # Filter out rows that are the last observations before a brake or a full stop for each permno
    crspm = crspm[crspm['month_diff']==1]
    
    # Only return the desired columns
    return crspm[['permno', 'date', 'altprc', 'excess_ret', 'excess_ret_ahead', 'me', 'yyyymm']]















        