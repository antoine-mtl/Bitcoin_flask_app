from flask import Flask, render_template, json,jsonify
import flask


import numpy as np
from datetime import timedelta
import pickle
import pandas as pd
import os
import json

# Graph
import plotly
import plotly.graph_objs as go


# LSTM Prediction
from sklearn.linear_model import LinearRegression
from keras import backend as K
from keras.models import load_model
import quandl



app = Flask(__name__)

@app.route('/')


def main():

    df_prediction_plot,bitcoin_data=bitcoin_pred(load_data=True)
    bitcoin_data=bitcoin_data.ix[-365:]
    bitcoin_data.columns=["date","open","high","low","close","volume","volume (Currency)","weighted Price"]
    print(bitcoin_data.head())
    bitcoin_data['ma1']=bitcoin_data.close.ewm(span=20, adjust=False).mean()

    bitcoin_data['ma3'] = bitcoin_data.close.ewm(span=50, adjust=False).mean()
    bitcoin_data['ma5'] = bitcoin_data.close.ewm(span=5, adjust=False).mean()
    bitcoin_data['ma12'] = bitcoin_data.close.ewm(span=12, adjust=False).mean()
    bitcoin_data['vol_ma1'] = bitcoin_data.volume.ewm(span=20, adjust=False).mean()
    rsi70 = np.repeat(70, [(len(bitcoin_data.close))], axis=0)
    rsi30 = np.repeat(30, [(len(bitcoin_data.close))], axis=0)
    baseligne=np.repeat(0, [(len(bitcoin_data.close))], axis=0)
    bitcoin_data['MACD'] = bitcoin_data.close.ewm(span=12, adjust=False).mean()-bitcoin_data.close.ewm(span=26, adjust=False).mean()
    bitcoin_data['MACD_signal_line'] = bitcoin_data.MACD.ewm(span=9, adjust=False).mean()

    def computeRSI(data, time_window):
        diff = data.diff(1).dropna()  # diff in one field(one day)

        # this preservers dimensions off diff values
        up_chg = 0 * diff
        down_chg = 0 * diff

        # up change is equal to the positive difference, otherwise equal to zero
        up_chg[diff > 0] = diff[diff > 0]

        # down change is equal to negative deifference, otherwise equal to zero
        down_chg[diff < 0] = diff[diff < 0]


        up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
        down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

        rs = abs(up_chg_avg / down_chg_avg)
        rsi = 100 - 100 / (1 + rs)
        return rsi
    bitcoin_data['RSI'] = computeRSI(bitcoin_data.close, 14)
    bitcoin_data = bitcoin_data.rename(columns={'close': 'Bitcoin close price'})
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bitcoin_data.index, y=bitcoin_data['Bitcoin close price'],
                              mode='lines',
                              line={"color": '#337ab7'},
                              # color="#FFCB9E",
                              name='Bitcoin close price'))

    fig2.add_trace(go.Scatter(x=bitcoin_data.index, y=bitcoin_data.ma1,
                              mode='lines',
                              line={"color": '#FFCB9E'},
                              #color="#FFCB9E",
                              name='Bitcoin 20 Day EMA'))


    fig2.add_trace(go.Scatter(x=bitcoin_data.index, y=bitcoin_data.ma5,
                              mode='lines',
                              line={"color": 'Violet'},
                              #color="Violet",
                              visible="legendonly",
                              name='Bitcoin 5 Day EMA'))
    fig2.add_trace(go.Scatter(x=bitcoin_data.index, y=bitcoin_data.ma12,
                              mode='lines',
                              line={"color": 'SlateBlue'},
                              # color="SlateBlue",
                              visible="legendonly",
                              name='Bitcoin 12 Day EMA'))
    fig2.add_trace(go.Scatter(x=bitcoin_data.index, y=bitcoin_data.ma3,
                              mode='lines',
                              line={"color": '#51B051'},
                              #color="#51B051",
                              name='Bitcoin 50 Day EMA'))

    fig2.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ))
    fig2.update_layout(
        title='Bitcoin price and Expential Moving Average (EMA)',
        autosize=False,
        width=1200,
        height=600,
        margin=dict(
            l=10,
            r=50,
            b=30,
            t=90,
            pad=4
        ))
    df_prediction_plot = df_prediction_plot.rename(columns={'yval': 'Bitcoin close price'})

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(x=df_prediction_plot.date, y=df_prediction_plot['Bitcoin close price'],
                              mode='lines',
                              line={"color": "#337ab7"},
                              # color="#FFCB9E",
                              name='Bitcoin close price'))
    fig3.add_trace(go.Scatter(x=df_prediction_plot.date, y=df_prediction_plot.yhat,
                              mode='lines',
                              line= {"color": "red", "dash": 'dot'},
                              # color="#FFCB9E",
                              name='Prediction Neural Net(LSTM)'))

    fig3.add_trace(go.Scatter(x=df_prediction_plot.date, y=df_prediction_plot.yhat_reg,
                              mode='lines',
                              line={"color": "red"},
                              # color="Violet",
                              name='Prediction regression'))
    fig3.update_layout(
        title='Bitcoin close price prediction',
        autosize=False,
        width=1240,
        height=600,
        margin=dict(
            l=10,
            r=50,
            b=30,
            t=30,
            pad=4
        ))
    bitcoin_data = bitcoin_data.rename(columns={'volume': 'Bitcoin 24H volume'})

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=bitcoin_data.date, y=bitcoin_data['Bitcoin 24H volume'], fill='tonexty',
                            line={"color": "#337ab7"},
                             mode='none',  # override default markers+lines
                             name='24 H Bitcoin volume '
                            ))
    fig4.add_trace(go.Scatter(x=bitcoin_data.date, y=bitcoin_data.vol_ma1,

                              line={"color": "#51B051"},
                              # color="#FFCB9E",
                              name='Bitcoin volume 20 Day EMA'))
    fig4.update_layout(
        title='Volume and Expential Moving Average (EMA)',
        autosize=False,
        width=600,
        height=400,
        margin=dict(
            l=10,
            r=100,
            b=30,
            t=30,
            pad=4
        ))

    fig5 = go.Figure()

    fig5.add_trace(go.Scatter(x=bitcoin_data.date, y=rsi70,

                              line={"color": "red", "dash": 'dot'},
                              name='Overbought (70)',
                              showlegend=True
                              ))
    fig5.add_trace(go.Scatter(x=bitcoin_data.date, y=bitcoin_data.RSI,
                              line={"color": "#337ab7"},
                              name='RSI (14 day)',
                              showlegend=True
                              ))
    fig5.add_trace(go.Scatter(x=bitcoin_data.date, y=rsi30,
                              line={"color": "red", "dash": 'dot'},
                              name='Oversell (30)',
                              showlegend=True
                              ))
    fig5.add_trace(go.Scatter(x=bitcoin_data.date, y=bitcoin_data.MACD,
                              line={"color": "#FFCB9E"},
                              name='MACD',
                              visible="legendonly"
                              ))
    fig5.add_trace(go.Scatter(x=bitcoin_data.date, y=bitcoin_data.MACD_signal_line,
                              line={"color": "#51B051"},
                              name='MACD Signal Line',
                              visible="legendonly"
                              ))
    fig5.add_trace(go.Scatter(x=bitcoin_data.date, y=baseligne,
                              line={"color": "red", "dash": 'dot'},
                              name='MACD base ligne',
                              visible="legendonly"
                              ))
    fig5.update_layout(
        title='RSI and MACD',
        autosize=False,
        width=580,
        height=400,
        margin=dict(
            l=50,
            r=45,
            b=30,
            t=30,
            pad=4
        ))


    graphs = [dict(data=fig2), dict(data=fig3), dict(data=fig4),
              dict(data=fig5)]
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    print(ids)
    return render_template('index.html',ids=ids,graphJSON=graphJSON)

def bitcoin_pred(load_data=False):
    # volume delta against previous day
    def get_DATA():

        def get_quandl_data(quandl_id):

            '''Download and cache Quandl dataseries'''
            cache_path = '{}.pkl'.format(quandl_id).replace('/', '-')

            try:
                os.remove(cache_path)
                print(cache_path)
                f = open(cache_path, 'rb')
                df = pickle.load(f)
                print('Loaded {} from cache'.format(quandl_id))
            except (OSError, IOError) as e:
                print('Downloading {} from Quandl'.format(quandl_id))
                df = quandl.get(quandl_id, authtoken='a3qYe_N9oy5Uba4d4x8c', returns="pandas")
                df.to_pickle(cache_path)
                print('Cached {} at {}'.format(quandl_id, cache_path))
            return df

        btc_usd_price_BITSTAMP = get_quandl_data('BCHARTS/KRAKENUSD')

        btc_usd_price_BITSTAMP.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volume (Currency)',
                                          'Weighted Price']
        btc_usd_price_BITSTAMP.insert(0, 'Date', btc_usd_price_BITSTAMP.index)
        taille = btc_usd_price_BITSTAMP.shape[0]
        btc_usd_price_BITSTAMP = btc_usd_price_BITSTAMP.iloc[taille - 1000:taille, :]
        print(btc_usd_price_BITSTAMP.shape)
        print('HEAD')
        btc_usd_price_BITSTAMP.to_csv("bitcoin.csv")

        return btc_usd_price_BITSTAMP

    if load_data:
        bitcoin_data = get_DATA()
        print('OK')
    else:
        bitcoin_data = pd.read_csv("bitcoin.csv")

    taille = bitcoin_data.shape[0]
    bitcoin_validation = bitcoin_data.iloc[taille - 20:taille, :]
    series = bitcoin_validation

    def predict_lstm(decalage, series):

        sc = pickle.load(open("sc.p", "rb"))
        train_sc = sc.transform(series.Close.values.reshape(-1, 1))

        X_train = train_sc[:-1]
        train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=series.index)
        print("PL2")
        for s in range(1, (decalage + 1)):
            train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)

        X_train = train_sc_df.dropna().drop('Scaled', axis=1)

        X_train = X_train.values

        X_train_t = X_train.reshape(X_train.shape[0], 1, decalage)

        K.clear_session()
        model = load_model('model_pred.h7')

        y_pred = model.predict(X_train_t)
        Y_pred_no_scale = sc.inverse_transform(y_pred)

        return Y_pred_no_scale

    data2 = {'date': bitcoin_validation.Date}

    df_prediction_plot = pd.DataFrame(data2)
    df_prediction_plot['date'] = pd.to_datetime(df_prediction_plot['date'])
    df_prediction_plot = df_prediction_plot.append(
        df_prediction_plot.iloc[df_prediction_plot['date'].shape[0] - 1] + timedelta(days=1))
    df_prediction_plot = df_prediction_plot.append(
        df_prediction_plot.iloc[df_prediction_plot['date'].shape[0] - 1] + timedelta(days=1))

    df_prediction_plot = df_prediction_plot.append(
        df_prediction_plot.iloc[df_prediction_plot['date'].shape[0] - 1] + timedelta(days=1))


    x = np.array([""])
    empty1 = np.repeat(x, [3], axis=0)
    empty2 = np.repeat(x, [(len(bitcoin_validation.Close) - 1)], axis=0)

    list_concatene = np.append(bitcoin_validation.Close, empty1)
    df_prediction_plot['yval'] = list_concatene

    last_value = bitcoin_validation.iloc[(len(bitcoin_validation.Close) - 1), 4]

    list_concatene2 = np.append(empty2, last_value)
    yp1 = predict_lstm(decalage=6, series=series)
    print(series.shape)
    yp1 = yp1[-1]

    data = {'Date': [yp1], 'Open': [yp1], 'High': [yp1], 'Low': [yp1], 'Close': [yp1], 'Volume': [yp1],
            'Volume (Currency)': [yp1], 'Weighted Price': [yp1]}
    df2 = pd.DataFrame(data)
    series = series.append(df2)
    print(series.shape)
    yp2 = predict_lstm(decalage=6, series=series)
    print(yp2)
    yp2 = yp2[-1]

    data = {'Date': [yp2], 'Open': [yp2], 'High': [yp2], 'Low': [yp2], 'Close': [yp2], 'Volume': [yp2],
            'Volume (Currency)': [yp2], 'Weighted Price': [yp2]}
    df3 = pd.DataFrame(data)
    series = series.append(df3)
    print(series.shape)
    yp3 = predict_lstm(decalage=6, series=series)
    print(yp3)
    yp3 = yp3[-1]

    list_concatene2 = np.append(list_concatene2, yp1)
    list_concatene2 = np.append(list_concatene2, yp2)
    list_concatene2 = np.append(list_concatene2, yp3)
    df_prediction_plot['yhat'] = list_concatene2

    model1 = LinearRegression()
    nbmax = len(bitcoin_data.Close)
    X1 = bitcoin_data.Close[(nbmax - 21):(nbmax - 1)]
    y = bitcoin_data.Close[(nbmax - 20):nbmax]

    model1.fit(X1.values.reshape(-1, 1), y.values.reshape(-1, 1))
    yp1 = model1.predict(bitcoin_data.Close[(nbmax - 1)].reshape(-1, 1))

    print('Ok')
    yp2 = model1.predict(yp1)
    yp3 = model1.predict(yp2)
    list_concatene3 = np.append(empty2, last_value)
    list_concatene3 = np.append(list_concatene3, yp1)
    list_concatene3 = np.append(list_concatene3, yp2)
    list_concatene3 = np.append(list_concatene3, yp3)
    df_prediction_plot['yhat_reg'] = list_concatene3

    return df_prediction_plot, bitcoin_data

@app.route('/.well-known/brave-rewards-verification.txt')


def file_sender():
    return flask.send_file("brave-rewards-verification.txt")


if __name__ == "__main__":
    app.run(port=5000)
