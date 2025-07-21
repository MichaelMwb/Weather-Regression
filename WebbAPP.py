from flask import Flask, request, render_template
import pandas as pd
from statsmodels.tsa.api import VAR
from datetime import timedelta
import logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logging.info('Upload route called')  # Log that the route was called
    if 'file' not in request.files:
        logging.error('No file part')  # Log an error message
        return render_template('home.html', error='No file part')
    f = request.files['file']
    if f.filename == '':
        logging.error('No selected file')  # Log an error message
        return render_template('home.html', error='No selected file')
    if f and f.filename.endswith('.csv'):
        try:
            df = pd.read_csv(f)
            logging.info(f'File uploaded: {f.filename}')  # Log the filename
            if 'DATE' not in df.columns:
                logging.error('No "DATE" column in CSV')  # Log an error message
                return render_template('home.html', error='No "DATE" column in CSV.')
            df['DATE'] = pd.to_datetime(df['DATE'])  # This will raise an exception if the DATE column cannot be converted to datetime
            logging.info('DATE column converted to datetime')  # Log a success message
            logging.info(df['DATE']) 
            station_name = df['NAME'].iloc[1]
        except Exception as e:
            return render_template('home.html', error='Error reading CSV: ' + str(e))
        print(station_name)
        print(type(station_name))

        last_date = df['DATE'].max()
        max_date = (last_date + timedelta(days=365)).strftime('%m-%d-%y')
        columns = df.columns.tolist()  # Get column names
        output_table = None
        #variable = request.form.get('selectedVariable')
        #variable = 'TMAX'
        #if not variable:
            #logging.error('No variable selected')  # Log an error message
            #return render_template('home.html', error='No variable selected')

        if request.form.get('forecastClicked') == 'true':
            
            start_date = pd.to_datetime(request.form.get('startDate'))  # Get the start date from the form
            end_date = pd.to_datetime(request.form.get('endDate'))  # Get the end date from the form
            future_dates = pd.date_range(start=start_date, end=end_date)
            variable = request.form.get('selectedVariable')
            logging.info(f'Selected variable: {variable}')  # Get the end date from the form
            print(type(station_name))
            logging.info(f'Selected variable: {variable}')
            logging.info(f'Start date: {start_date}')
            logging.info(f'End date: {end_date}')
            logging.info(f'Form data: {request.form}')
            try:
                date_column_index = df.columns.get_loc('DATE')
                print(date_column_index)  # Get the index of the 'DATE' column
            except Exception as e:
                print(f"An error occurred: {e}")
            date_column_index = df.columns.get_loc('DATE')
            print(date_column_index)  # Get the index of the 'DATE' column
            df_selected = df.iloc[:, date_column_index+1:] 
            df_selected = df_selected.fillna(0)  # Fill NaN values with 0
            model = VAR(df_selected)  # Use selected columns to train the model
            model_fit = model.fit(maxlags=5, ic='aic')  # Adjust the maxlags parameter as needed
            last_observations = df_selected.values[-model_fit.k_ar:]

            forecast_result = model_fit.forecast(last_observations, steps=len(future_dates))

            forecast_df = pd.DataFrame(forecast_result, index=future_dates, columns=df_selected.columns)
            forecast_var = forecast_df[variable]

# Create 'DATE' column from index and convert to datetime
            forecast_df['DATE'] = pd.to_datetime(forecast_df.index)

# Reformat 'DATE' to 'mm/dd/yyyy'
            forecast_df['DATE'] = forecast_df['DATE'].dt.strftime('%m/%d/%Y')
            
            
            forecast_df = pd.DataFrame({
                'DATE': forecast_df['DATE'],
                variable: forecast_var
    })
            column_mapping = {
                'PRCP': 'Precipitation (in)',
                'SNOW': 'Snowfall (in)',
                'TAVG': 'Average Temperature (°F)',
                'TMAX': 'Maximum Temperature (°F)',
                'TMIN': 'Minimum Temperature (°F)',
                'AWND': 'Average Wind Speed (mph)'
            }  # Add your desired columns here

            forecast_df.rename(columns=column_mapping, inplace=True)


    logging.info(f'Type of df: {type(df)}')
    logging.info(f'Type of forecast_df: {type(forecast_df)}')

    df = pd.concat([df, forecast_df])

    if forecast_df.empty:
        logging.info('forecast_df is empty')
    else:
       output_table = forecast_df.to_html(index=False)

    return render_template('home.html', error=None, max_date=max_date, columns=columns[2:], station_name=station_name,output_table=output_table)
  

if __name__ == '__main__':
    app.run(debug=True)