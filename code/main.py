import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
import sqlite3

conn = sqlite3.connect('/TheIntelligentInvestor/data/data.db')
c = conn.cursor()


# Functions
def get_stock_by_id(stock_id):
    c.execute('SELECT * FROM owned_stocks WHERE stock_id=:stock', {"stock": stock_id})
    data = c.fetchall()
    return data


def get_all_stocks():
    c.execute('SELECT * FROM owned_stocks')
    data = c.fetchall()
    return data


def add_stock(date_purchase, value_purchase, stock_id):
    c.execute('INSERT INTO owned_stocks(date_purchase,value_purchase,stock_id) VALUES (?,?,?)',
              (date_purchase, value_purchase, stock_id))
    conn.commit()


def main():
    # Layout Templates
    html_temp = """
    <div style="background-color:{};padding:10px;border-radius:10px">
    <h1 style="color:{};text-align:center;">The Intelligent Investor  </h1>
    </div>
    """
    """A Simple Stock Portfolio Visualizer"""

    st.markdown(html_temp.format('royalblue', 'white'), unsafe_allow_html=True)

    menu = ["Home", "Search Stocks", "View Portfolio"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("Home")
        st.write("Welcome to The Intelligent Investor app. You can use this application to save your purchased stocks, visualize your data and make predictions for instance.")


    elif choice == "Search Stocks":
        st.subheader("Buy Stock")
        start = "2015-01-01"
        today = date.today().strftime("%Y-%m-%d")
        data_load_state = st.text('Loading data...')

        @st.cache
        def load_data(ticker):
            data = yf.download(ticker, start, today)
            data.reset_index(inplace=True)
            data["Date"] = pd.to_datetime(data['Date']).dt.date
            return data

        stocks = ("AAPL", "MSFT", "GME")
        selected_stock_id = st.selectbox("Select Stock Id", stocks)
        data = load_data(selected_stock_id)
        data_load_state.text('Loading data... done!')
        #
        st.subheader('Raw data')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        if st.button("Buy"):
            last_obs = len(data["High"]) - 1
            value = (data["High"][last_obs] + data["Low"][last_obs]) / 2
            # nb_stocks = st.text_input("Enter Number of Stocks", max_chars=50)
            add_stock(data["Date"][last_obs], value, selected_stock_id)
            st.success("Post:{} saved".format(selected_stock_id))

        if st.checkbox("History"):
            st.subheader("Stock historical data")
            plot_raw_data()

        if st.checkbox("Predictions"):
            st.subheader("FB prophet quantitative predictor")
            #
            n_months = st.slider("Months of prediction:", 1, 4)
            period = n_months * 30
            #
            # Predict forecast with Prophet.
            df_train = data[['Date', 'Close']]
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
            #
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)
            #
            # Show and plot forecast
            st.subheader('Forecast data')
            st.write(forecast.tail())
            #
            st.write(f'Forecast plot for {n_months} months')
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)
            #
            st.write("Further forecasting analysis")
            fig2 = m.plot_components(forecast)
            st.write(fig2)

    elif choice == "View Portfolio":
        st.subheader("View Portfolio")
        start = "2015-01-01"
        today = date.today().strftime("%Y-%m-%d")
        data_load_state = st.text('Loading data...')

        @st.cache
        def load_data(ticker):
            data = yf.download(ticker, start, today)
            data.reset_index(inplace=True)
            data["Date"] = pd.to_datetime(data['Date']).dt.date
            return data

        stocks = ("AAPL", "MSFT", "GME")
        selected_stock_id = st.selectbox("Select The Owned Stock", stocks)
        data = load_data(selected_stock_id)
        data_load_state.text('Loading data... done!')
        #
        st.subheader('Raw data')
        owned_stocks_raw = pd.DataFrame(get_all_stocks())
        st.write(owned_stocks_raw)
        names = owned_stocks_raw[2].unique()
        if owned_stocks_raw.loc[owned_stocks_raw[2] == selected_stock_id][0].ndim == 1:
            ess = owned_stocks_raw.loc[owned_stocks_raw[2] == selected_stock_id][0][0]
            bole = data["Date"] > date(int(ess[0:4]), int(ess[5:7]), int(ess[8:10]))
        else:
            ess = owned_stocks_raw.loc[owned_stocks_raw[2] == selected_stock_id][0][1]
            bole = data["Date"] > date(int(ess[0:4]), int(ess[5:7]), int(ess[8:10]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.loc[bole]['Date'], y=data.loc[bole]['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data.loc[bole]['Date'], y=data.loc[bole]['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        if st.checkbox("Total"):
            total = owned_stocks_raw.groupby(by=2).sum()
            st.write(total)
            piefig = px.pie(values=total[1], names=names,
                            color_discrete_sequence=px.colors.sequential.dense)
            st.plotly_chart(piefig)


if __name__ == '__main__':
    main()
