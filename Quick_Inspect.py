import pandas as pd

def get_endpoint(ticker_request):
    endpoint_data = f"https://raw.githubusercontent.com/alitalbi/storage_data_fy/refs/heads/master/{ticker_request}.csv"
    return endpoint_data


def quick_inspect(file_path):
    """Quick snippet of data strcuture"""
    print("=" * 50)
    print("Quick Data Inspection")
    print("=" * 50)

    try:
        input_price_data = pd.read_csv(file_path)
    except:
        input_price_data = pd.DataFrame()
        print("Importing failed. Please check file path or file format.")

    print(f"Data Shape : {input_price_data.shape}")
    print(f"Columns :\n{input_price_data.columns.tolist()}")
    print(f"Data Types : \n{input_price_data.dtypes}")
    print(f"First 3 rows : \n{input_price_data.head(3)}")
    print(f"Last 3 rows : \n{input_price_data.tail(3)}")
    print(f"Basic stats : \n{input_price_data.describe()}")
    print(f"Missing Values : \n{input_price_data.isnull().sum()}")
    print(f"Memory Usage : {input_price_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    return input_price_data

file_path = get_endpoint("GC=F")
imported_backtest_data = quick_inspect(file_path)
