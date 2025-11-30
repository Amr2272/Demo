# [file name]: predictions.py
# [file content begin]
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, Optional
from prediction_logger import PredictionLogger

# Initialize the logger
prediction_logger = PredictionLogger()

def batch_predict_mlflow(model, model_type, end_date: date, prophet_df=None) -> pd.DataFrame:
    """Generate batch predictions until a specific end date using MLflow model."""
    try:
        # Calculate periods based on end date
        if prophet_df is not None and not prophet_df.empty:
            last_data_date = prophet_df['ds'].max()
        else:
            last_data_date = pd.Timestamp(datetime.now().date())
        
        end_date_dt = pd.Timestamp(end_date)
        periods = (end_date_dt - last_data_date).days
        
        if periods <= 0:
            st.warning("End date must be after the last data date.")
            return pd.DataFrame()
        
        if model_type == "prophet":
            # For Prophet models, we need to create future dataframe
            future = pd.DataFrame({
                'ds': pd.date_range(start=last_data_date + timedelta(days=1), 
                                  periods=periods, freq='D')
            })
            forecast = model.predict(future)
            
            # Log the predictions - ONLY TO MLFLOW
            log_count = prediction_logger.log_batch_prediction(
                forecast_data=forecast.rename(columns={'ds': 'date', 'yhat': 'prediction'}),
                actual_data=prophet_df,
                model_type=model_type,
                forecast_end_date=end_date
            )
            
            st.info(f"📊 Logged {log_count} predictions to MLflow")
            st.success("✅ Predictions successfully logged to MLflow")
            
            return forecast
            
        elif model_type == "arima":
            # For ARIMA models - create proper input format
            future_dates = pd.date_range(start=last_data_date + timedelta(days=1), periods=periods, freq='D')
            
            # Create input with date information that matches the wrapper's expected schema
            future_df = pd.DataFrame({
                'ds': future_dates,
                'year': future_dates.year.astype('int32'),
                'month': future_dates.month.astype('int32'),
                'day': future_dates.day.astype('int32')
            })
            
            # Get predictions
            predictions_df = model.predict(future_df)
            
            # Create forecast DataFrame
            forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': predictions_df['prediction'].values if 'prediction' in predictions_df.columns else predictions_df.iloc[:, 0].values
            })
            
            # Log the predictions - ONLY TO MLFLOW
            log_count = prediction_logger.log_batch_prediction(
                forecast_data=forecast.rename(columns={'ds': 'date', 'yhat': 'prediction'}),
                actual_data=prophet_df,
                model_type=model_type,
                forecast_end_date=end_date
            )
            
            st.info(f"📊 Logged {log_count} predictions to MLflow")
            st.success("✅ Predictions successfully logged to MLflow")
            
            return forecast
            
        elif model_type == "lightgbm":
            # For LightGBM models - create feature dataframe with ALL required features
            future_dates = pd.date_range(start=last_data_date + timedelta(days=1), periods=periods, freq='D')
            
            # Create input with ALL feature columns that the model expects
            future_df = pd.DataFrame({
                'ds': future_dates,
                'year': future_dates.year.astype('float64'),
                'month': future_dates.month.astype('float64'),
                'day': future_dates.day.astype('float64'),
                'dayofweek': future_dates.dayofweek.astype('float64'),
                'quarter': future_dates.quarter.astype('float64'),
                'dayofyear': future_dates.dayofyear.astype('float64'),
                'weekofyear': future_dates.isocalendar().week.astype('float64')
            })
            
            try:
                # Get predictions
                predictions_df = model.predict(future_df)
                
                forecast = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': predictions_df['prediction'].values
                })
                
                # Log the predictions - ONLY TO MLFLOW
                log_count = prediction_logger.log_batch_prediction(
                    forecast_data=forecast.rename(columns={'ds': 'date', 'yhat': 'prediction'}),
                    actual_data=prophet_df,
                    model_type=model_type,
                    forecast_end_date=end_date
                )
                
                st.info(f"📊 Logged {log_count} predictions to MLflow")
                st.success("✅ Predictions successfully logged to MLflow")
                
                return forecast
                
            except Exception as e:
                st.error(f"LightGBM prediction failed: {e}")
                return pd.DataFrame()
            
        else:
            st.error(f"Unsupported model type: {model_type}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error in batch prediction: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()
    
def real_time_predict_mlflow(model, model_type, target_date: datetime, context_data: Optional[Dict] = None, prophet_df=None) -> Dict:
    """Generate prediction for a single specific date using MLflow model."""
    try:
        target_date_dt = pd.to_datetime(target_date)
        
        if model_type == "prophet":
            future_df = pd.DataFrame({'ds': [target_date_dt]})
            forecast = model.predict(future_df)
            
            result = {
                'date': target_date,
                'prediction': float(forecast['yhat'].iloc[0]),
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            }
            
        elif model_type == "arima":
            # For ARIMA, we need to generate a sequence of predictions and take the last one
            # Calculate how many days from the last training data to the target date
            if prophet_df is not None and not prophet_df.empty:
                last_data_date = prophet_df['ds'].max()
            else:
                last_data_date = pd.Timestamp(datetime.now().date())
            
            days_ahead = (target_date_dt - last_data_date).days
            
            if days_ahead <= 0:
                result = {
                    'date': target_date,
                    'prediction': 0,
                    'model_type': model_type,
                    'error': 'Target date must be after the last training data date',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Create input for the required number of periods
                future_dates = pd.date_range(start=last_data_date + timedelta(days=1), periods=days_ahead, freq='D')
                
                future_df = pd.DataFrame({
                    'ds': future_dates,
                    'year': future_dates.year.astype('int32'),
                    'month': future_dates.month.astype('int32'),
                    'day': future_dates.day.astype('int32')
                })
                
                # Get predictions for all days up to the target date
                predictions_df = model.predict(future_df)
                
                # Take the last prediction (the one for our target date)
                if 'prediction' in predictions_df.columns:
                    prediction_value = float(predictions_df['prediction'].iloc[-1])
                else:
                    prediction_value = float(predictions_df.iloc[-1, 0])
                
                result = {
                    'date': target_date,
                    'prediction': prediction_value,
                    'model_type': model_type,
                    'timestamp': datetime.now().isoformat()
                }
            
        elif model_type == "lightgbm":
            # LightGBM single prediction - create ALL required features
            lightgbm_input = pd.DataFrame({
                'ds': [target_date_dt],
                'year': np.array([target_date_dt.year], dtype='float64'),
                'month': np.array([target_date_dt.month], dtype='float64'),
                'day': np.array([target_date_dt.day], dtype='float64'),
                'dayofweek': np.array([target_date_dt.dayofweek], dtype='float64'),
                'quarter': np.array([target_date_dt.quarter], dtype='float64'),
                'dayofyear': np.array([target_date_dt.dayofyear], dtype='float64'),
                'weekofyear': np.array([target_date_dt.isocalendar().week], dtype='float64')
            })
            
            prediction_df = model.predict(lightgbm_input)
            
            result = {
                'date': target_date,
                'prediction': float(prediction_df['prediction'].iloc[0]),
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            }
            
        else:
            result = {
                'date': target_date,
                'prediction': None,
                'model_type': model_type,
                'error': 'Unsupported model type',
                'timestamp': datetime.now().isoformat()
            }
            
        return result
        
    except Exception as e:
        return {
            'date': target_date,
            'prediction': None,
            'model_type': model_type,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def standardize_forecast_data(forecast_data, model_type):
    """Standardize forecast data to have consistent column names."""
    if forecast_data is None or forecast_data.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying original
    result = forecast_data.copy()
    
    # Handle different column naming conventions
    if 'ds' in result.columns and 'yhat' in result.columns:
        # Prophet format
        result = result.rename(columns={'ds': 'date', 'yhat': 'prediction'})
    elif 'prediction' in result.columns and 'date' not in result.columns:
        # Add date column if missing
        if 'ds' in result.columns:
            result = result.rename(columns={'ds': 'date'})
        elif len(result) > 0:
            # Create date range if no date column exists
            result['date'] = pd.date_range(start=datetime.now(), periods=len(result), freq='D')
    
    # Ensure we have the required columns
    if 'date' not in result.columns:
        st.error(f"Missing 'date' column in {model_type} forecast data")
        return pd.DataFrame()
        
    if 'prediction' not in result.columns:
        # Look for alternative prediction column names
        possible_pred_cols = ['yhat', 'forecast', 'y_pred', 'sales_pred']
        for col in possible_pred_cols:
            if col in result.columns:
                result = result.rename(columns={col: 'prediction'})
                break
        
        if 'prediction' not in result.columns and len(result.columns) > 1:
            # Use the first numeric column as prediction
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                result = result.rename(columns={numeric_cols[0]: 'prediction'})
            else:
                st.error(f"No prediction column found in {model_type} forecast data")
                return pd.DataFrame()
    
    return result[['date', 'prediction']]
# [file content end]