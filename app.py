import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import pandas as pd

from gsheets import get_spreadsheet
from model import train_model


# Definitions of constants. This projects uses extra CSS stylesheet at `./assets/style.css`
COLORS = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', '/assets/style.css']


# Define the dash app first
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


# Define component functions


def page_header():
    """
    Returns the page header as a dash `html.Div`
    """
    return html.Div(id='header', children=[
        html.Div([html.H3('Charting The Charts')],
                 className="ten columns"),
        html.A([html.Img(id='logo', src=app.get_asset_url('github.png'),
                         style={'height': '35px', 'paddingTop': '7%'}),
                html.Span('Github Repo', style={'fontSize': '2rem', 'height': '35px', 'bottom': 0,
                                                'paddingLeft': '4px', 'color': '#a3a7b0',
                                                'textDecoration': 'none'})],
               className="two columns row",
               href='https://github.com/ncl11/1050_project'),
    ], className="row")


def description():
    """
    Returns overall project description in markdown
    """
    return html.Div(children=[dcc.Markdown('''
        # About
        ### Project & Executive Summary
        It is no secret that the movie industry has begun to tap into immensely effective techniques through social media and news articles to create conversation online about upcoming movie releases. But how effective is the phenomenon of “online buzz”?

        The recent trend of multi-million blockbuster movies (the annual MCU summer releases come to mind), which were prefaced with non-stop online conversation for months, sometimes years, underlines the idea that a big part of what makes a movie money is how many people can be convinced to watch it. And statistically, if more people are talking about a movie, it will make more people curious enough to go see the movie.

        Charting the Charts is a tool to utilize the potential predictive power of online trending data to predict future revenue trends for new movie releases.
        
        The data used in this model is retrieved from two data sources. The BoxOfficeMojo data is scraped and the Google Trends data is available through gtab.  Our database updates automatically every Tuesday with the previous week of data.  Unsurprisingly the most predictive variable for box office gross was the gross from the previous week.  This correlation can be seen below.

        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")


COLORS = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']
c = ['red', 'blue', 'orange', 'white']
def static_stacked_trend_graph(stack=False):
    """
    Returns scatter plot.

    """

    trends = []
    revenues = []
    
    df = get_spreadsheet()
    df['Date_dt'] = df['Date'].astype('datetime64[ns]')
    date = df['Date_dt'].iloc[-1]

    stack = False
    if df is None:
        return go.Figure()
    
    fig = go.Figure()
    movie_list = df[df['Date_dt'] == date].sort_values(by=['Weekly'], ascending=False)['Release'][0:20]
    for row in range(df.shape[0]):
        date = df['Date_dt'].iloc[-1]

        trend = df.iloc[row]['Weekly']

        rev = df.iloc[row]['Week + 1']
        if rev == 0:
            continue

        trends.append(trend)
        revenues.append(rev)
        fig.add_trace(go.Scatter(x=[trend], y=[rev], mode='markers', name=df.iloc[row]["Release"],
                         line={'width': 2, 'color': c[row%4]},
                         stackgroup='stack' if stack else None))
    trends, revenues = pd.Series(trends), pd.Series(revenues)
    corr = trends.corr(revenues)
    
    title = f'Weekly Gross(week i) vs Weekly Gross(week i+1): Correlation = {corr}'

    fig.update_layout(template='plotly_dark',
                      title=title,
                      plot_bgcolor='#23272c',
                      paper_bgcolor='#23272c',
                      yaxis_title='Weekly Gross(week i+1)',
                      xaxis_title='Weekly Gross(week i)')
    return fig

def description2():
    """
    Returns overall project description in markdown
    """
    return html.Div(children=[dcc.Markdown('''
    The correlation between Google Trend data and gross varies widely from week to week.  The data was highly correlated with the gross for some weeks and not strongly correlated for others.  This is largely dependent on the names of the highest grossing movies and if there is overlap with common search words on Google.  Below are two stacked bar plots of the normalized Gross and Google Trend data for the week beginning 6/22/20 and the most current week respectively.  Some of the issues that arise when using Google Trend data are discussed below in the Next Steps section. 

        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")

def static_stacked_bar_graph(stack=False):
    df = pd.read_csv('6_mo_weekly.csv', sep='\t')
    df['Date_dt'] = df['Date'].astype('datetime64[ns]')

    date = df['Date_dt'].unique()[5]

    fig = go.Figure()
    sorted_df = df[df['Date_dt'] == date].sort_values(by=['Weekly'], ascending=False)[0:15][::-1]
    y = sorted_df['Release']
    x_rev = sorted_df['Weekly'] / np.sum(sorted_df['Weekly'])
    x_trend = sorted_df['google trends'] / np.sum(sorted_df['google trends'])
    trends, revenues = pd.Series(x_trend), pd.Series(x_rev)
    corr = trends.corr(revenues)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y,
        x=x_rev,
        name='Gross',
        orientation='h',
        marker=dict(
            color='blue', #rgba(246, 78, 139, 0.6)',
            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
        )
    ))
    fig.add_trace(go.Bar(
        y=y,
        x=x_trend,
        name='Google Trends',
        orientation='h',
        marker=dict(
            color='red', #rgba(58, 71, 80, 0.6)',
            line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
        )
    ))
    title = f'Normalized Gross and Google Trends for top Movies-Week Starting 6/22/2020: Correlation = {corr}'

    fig.update_layout(template='plotly_dark',
                    title=title,
                    plot_bgcolor='#23272c',
                    paper_bgcolor='#23272c',
                    yaxis_title='Movies',
                    xaxis_title='Normalized Quantities',barmode='stack')


    return fig
    
def static_stacked_bar_graph_current(stack=False):
    df = get_spreadsheet()
    df['Date_dt'] = df['Date'].astype('datetime64[ns]')

    date = df['Date_dt'].iloc[-1]

    fig = go.Figure()
    sorted_df = df[df['Date_dt'] == date].sort_values(by=['Weekly'], ascending=False)[0:15][::-1]
    y = sorted_df['Release']
    x_rev = sorted_df['Weekly'] / np.sum(sorted_df['Weekly'])
    x_trend = sorted_df['google trends'] / np.sum(sorted_df['google trends'])
    trends, revenues = pd.Series(x_trend), pd.Series(x_rev)
    corr = trends.corr(revenues)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y,
        x=x_rev,
        name='Gross',
        orientation='h',
        marker=dict(
            color='blue', #rgba(246, 78, 139, 0.6)',
            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
        )
    ))
    fig.add_trace(go.Bar(
        y=y,
        x=x_trend,
        name='Google Trends',
        orientation='h',
        marker=dict(
            color='red', #rgba(58, 71, 80, 0.6)',
            line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
        )
    ))
    title = f'Normalized Gross and Google Trends for top 15 Movies-Most Recent Data: Correlation = {corr}'

    fig.update_layout(template='plotly_dark',
                    title=title,
                    plot_bgcolor='#23272c',
                    paper_bgcolor='#23272c',
                    yaxis_title='Movies',
                    xaxis_title='Normalized Quantities',barmode='stack')


    return fig


def what_if_description():
    """
    Returns description of the prediction plot
    """
    return html.Div(children=[
        dcc.Markdown('''
        # Our Model
        Below we have a plot of the top 5 movies from the previous week and our prediction for the upcoming week of gross.  The chart is updated automatically with new gross data and a new prediction every week as new data comes in.  The top slider allows the user to toggle through movies 1 to 5 and the second binary slider shows our prediction.  We have found Random Forest Regression model to be very effective.  Please note that each data point/date corresponds with data recorded for the week which begins on that date and likewise for the predictions.  For example gross shown with a date of 11/30/2020 is actually data from 11/30-12/6'''
    )], className="row")


def what_if_tool():
    """
    Returns prediction plot as a dash `html.Div`. 
    """
    return html.Div(children=[
        html.Div(children=[dcc.Graph(id='what-if-figure')], className='nine columns'),

        html.Div(children=[
            html.H5("Current Top 5 Movies", style={'marginTop': '2rem'}),
            html.Div(children=[
                dcc.Slider(id='movie-slider', min=0, max=4, step=1, value=0, className='row',
                           marks={x: str(x) for x in np.arange(0, 4.1, 1)})
            ], style={'marginTop': '5rem'}),

            html.Div(id='movie-text', style={'marginTop': '1rem'}),

            html.Div(children=[
                dcc.Slider(id='pred-slider', min=0, max=1, step=1, value=0,
                           className='row', marks={x: str(x) for x in np.arange(0, 1.1, 1)})
            ], style={'marginTop': '3rem'}),
            html.Div(id='pred-text', style={'marginTop': '1rem'}),
        ], className='three columns', style={'marginLeft': 5, 'marginTop': '10%'}),
    ], className='row eleven columns')


def architecture_summary():
    """
    Returns next steps, additional info and other text
    """
    return html.Div(children=[
        dcc.Markdown('''
            # Additional Information

            When using Google trends data we encountered several problems.  The first is that often times the name of a movie can overlap with common search words on Google.  In this case the Google trends data will be greatly over exagerated.  For example the movie 'Tesla' had some of the highest Google trends data for any movie but not especially high revenue.  This is because Tesla is a very successful company as well as a movie, so many of the searches presumably were related to the company.  In order to mitigate this issue we add the word 'movie' to each of the movie titles that we search for.  Conversely, we would also frequently see movies with very lengthy names return very low number for Google trends.  This is likely because people wouldn't search for the entire title.  We were able to partially mitigate this problem by removing part of the title for any movies that fell into one of two categories.  The first category contained movies with a colon.  For example, in the case of the movie 'The Spongebob Movie: Sponge on the Run' we would just search for 'The Spongebob Movie'.  The second category contained movies with the text '2020 Re-release'.  For example, for the movie 'Elf 2020 Re-release' we would just search for 'Elf movie'.

            After trying several models we ultimately found that Random Forest Regression performed optimally but we also experimented with Ridge Regression, K-Nearest Neighbors Regression and Support Vector Regression.  We feel the model could be improved through more accurate data and additional features.  We frequently encountered issues/errors with gtab when retrieving Google trends data so we would see variation on data for a given movie for identical search queries.  We would also like to incorporate data from twitter, instagram, facebook and other social media platforms.  Ideally we would like data that would allow us to quantify trends and sentiment on each platform.  

            ### Datasets Used 
            We acquired data from the following sites:

            * https://trends.google.com/trends/?geo=US 
            * https://www.boxofficemojo.com/?ref_=bo_nb_da_mojologo
            
            We obtained data from BoxOfficeMojo and daily searches from Google Trends and merged the two data sets on date and movie title.  Our dataset will be automatically updated at weekly intervals through rescraping BoxOfficeMojo and merging with the new Google Trends data. 

            ### ETL Processing, Database Design and Caching

            Extract - Movie title, date, percent change from previous day, percent change from previous week, number of days since release, per theatre average gross and to date gross was extracted by scraping BoxOfficeMojo and google trends numbers were retrieved using gtab. 

            Transform - Once the data was scraped, the resulting data frames were merged so that one data frame contained revenue and trend information for a specific movie. Data was stripped of extraneous characters, and converted into workable types and converted from daily to weekly data.  As a last step, for each date and movie we merged the gross for the following week as our target variable.    

            Load/Descriptor of Database Used -  Given the relatively small dataset in this problem, we have elected to host our data directly on Heroku.  Our cached dataset contains data from 8/24/2020 through the most recent week.  On Tuesday the app automatically updates with data from monday of the previous week through Sunday. We have elected to update on Tuesday because gtab data is updated two days after the fact so pulling on Monday returns no data for Sunday.  

            Code for the layout of the app has been adapted from: https://github.com/BrownDSI/data1050-demo-project-f20

        ''', className='row eleven columns', style={'paddingLeft': '5%'}),


        dcc.Markdown('''
        
        ''')
    ], className='row')


# Sequentially add page components to the app's layout
def dynamic_layout():
    return html.Div([
        page_header(),
        html.Hr(),
        description(),
        dcc.Graph(id='stacked-trend-graph', figure=static_stacked_trend_graph(stack=True)),
        description2(), 
        dcc.Graph(id='stacked-trend-graph2', figure=static_stacked_bar_graph(stack=False)),
        dcc.Graph(id='stacked-trend-graph3', figure=static_stacked_bar_graph_current(stack=False)),
        what_if_description(),
        what_if_tool(),
        architecture_summary(),
    ], className='row', id='content')


# set layout to a function which updates upon reloading
app.layout = dynamic_layout


# Defines the dependencies of interactive components

@app.callback(
    dash.dependencies.Output('movie-text', 'children'),
    [dash.dependencies.Input('movie-slider', 'value')])
def update_movie_text(value):
    """Changes the display text of the slider"""
    new_val = int(value) + 1
    return "Movie {:.2f}".format(new_val)


@app.callback(
    dash.dependencies.Output('pred-text', 'children'),
    [dash.dependencies.Input('pred-slider', 'value')])
def update_pred_text(value):
    """Changes the display text of the slider"""
    if value == 1:
        toggle = 'on'
    else:
        toggle = 'off'
    return f"Show Prediction -- On/Off: {toggle}"



@app.callback(
    dash.dependencies.Output('what-if-figure', 'figure'),
    [dash.dependencies.Input('movie-slider', 'value'),
     dash.dependencies.Input('pred-slider', 'value')])
def what_if_handler(movie, pred):
    stack = False
    df = get_spreadsheet()
    df['Date_dt'] = df['Date'].astype('datetime64[ns]')
    date = df['Date_dt'].iloc[-1]

    df_test = df.groupby('Release').filter(lambda x : x['Release'].shape[0]>=2)
    movie_list = df_test[df_test['Date_dt'] == date].sort_values(by=['Weekly'], ascending=False)['Release']

    movie = int(movie)
    pred = int(pred)
    df = df[df['Release'] == movie_list.iloc[movie]]
    if df is None:
        return go.Figure()
    x = df['Date']

    predict_date = df['Date_dt'].iloc[-1]  + timedelta(days=7)
    y_pred = train_model().predict([df.drop(['Release', 'Date', 'Y', 'Week + 1', 'Date_dt'], axis=1).iloc[-1]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df['Weekly'], mode='lines', name=movie_list.iloc[movie],
                             line={'width': 2, 'color': 'orange'},
                             stackgroup='stack' if stack else None))
    if pred:
        fig.add_trace(go.Scatter(x=[predict_date], y=y_pred, mode='markers', name='Prediction',
                                line={'width': 2, 'color': 'red'},
                                stackgroup='stack' if stack else None))
                                
    fig.update_layout(yaxis=dict(range=[0, 1.2*df['Weekly'].max()]), xaxis=dict(range=[x.iloc[0], predict_date+timedelta(days=1)]))


    title = f'Weekly Revenue for {movie_list.iloc[movie]}'

    fig.update_layout(template='plotly_dark',
                      title=title,
                      plot_bgcolor='#23272c',
                      paper_bgcolor='#23272c',
                      yaxis_title='MW',
                      xaxis_title='Date/Time')


    return fig


if __name__ == '__main__':
    app.run_server(debug=True) #, port=1050, host='0.0.0.0')
