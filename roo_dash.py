import pygsheets
import pandas as pd
import numpy as np
import datetime as dt
from datetime import date
import plotly.express as px
import dash
from dash import dcc
from dash import html
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#authorization
# echo $(pwd)/filename.xxx will reveal full path in mac terminal

# Path to api key on Roo's mac
path = '/Users/roo/Desktop/Python/Roo Project/JobHunting/roo-jobs-8d5b5646e1c4.json'

# Path to api key on pythonanywhere site
path = '/home/roophillips/mysite/roo-jobs-8d5b5646e1c4.json'

gc = pygsheets.authorize(service_account_file = path)

# Define functions for column index: Letter to Number and vice versa:
excel_col_name = lambda n: '' if n <= 0 else excel_col_name(
    (n - 1) // 26) + chr((n - 1) % 26 + ord('A'))

excel_col_num = lambda a: 0 if a == '' else 1 + ord(
    a[-1]) - ord('A') + 26 * excel_col_num(a[:-1])

# Open Roo's Job Tracker and do initial formatting
sh = gc.open('Job Tracker')
sheet1 = sh[0] # Select first tab
last_col = excel_col_name(sheet1.cols) # Identify number of columns
last_cell = str(last_col) + str(sheet1.rows+1) # Identify bottom right cell (range)
header = tuple(sheet1.get_row(1)) # Create header tuple
app_date_index = header.index('Application Date') # Find out index of app dat column 

# Create dataframe

# Make first row header, then drop the first row
df = pd.DataFrame(sheet1, 
                  columns = header).drop(index=0).reset_index().drop(['index'],axis=1)

# Create new df that fills empty strings (excel blanks) with NaNs
df2 = df.applymap(
    lambda x: np.nan if isinstance(x, str) and (not x or x.isspace()) else x)

# Change date columns to datatype date
df2['Application Date'] = pd.to_datetime(df2['Application Date'])
df2['Rejection Date'] = pd.to_datetime(df2['Rejection Date'])
df2['Interview 1 Date'] = pd.to_datetime(df2['Interview 1 Date'])
df2['Interview 2 Date'] = pd.to_datetime(df2['Interview 2 Date'])
df2['No Jobs as of Date'] = pd.to_datetime(df2['No Jobs as of Date'])

# Change numbered columns to datatype int
df2['Pay Low'] = pd.to_numeric(df2['Pay Low'])
df2['Pay High'] = pd.to_numeric(df2['Pay High'])

# Extract the dates that applications were submitted
app_dates = df2.groupby('Application Date', as_index=False).count()['Application Date']
app_dates = pd.DataFrame(app_dates)

# # Extract the sum of applications by date
app_count = df2.groupby('Application Date', as_index=False)['Application Date'].count()
app_count.columns = ['Submissions']

# # Extract the sum of rejections by date
app_reject = df2.groupby('Rejection Date', as_index=False)['Rejection Date'].count()
app_reject.columns = ['Rejections']

# New dataframe with columns grouped by date
df3=pd.concat([app_dates, app_count, app_reject], axis = 1)

# Create cumulitive sums
sub_cum = pd.DataFrame(np.cumsum(df3['Submissions']))
rej_cum = pd.DataFrame(np.cumsum(df3['Rejections']))

# Add cumulitive sums to dataframe
sub_cum.columns = ['Cumulative Submissions']
rej_cum.columns = ['Cumulative Rejections']
df4=pd.concat([df3, sub_cum, rej_cum], axis = 1)

# Pay statistics and charts
low_pay_med = int(df2['Pay Low'].median())
high_pay_med = int(df2['Pay High'].median())

# Days looking for a job
today = dt.datetime.now().replace(second=0, microsecond=0)
start_day = dt.datetime(2023,1,26)

days_looking = (today-start_day).days
df_days_looking = pd.DataFrame([{'Days on the market': days_looking}])

# Number of applications submitted
napps_sum = df2['Application Date'].count()

# Days between application date and rejection date
mask1 = df2['Rejection Date'].notnull()
df5 = df2[mask1][['Application Date', 'Rejection Date']]

df5['Rejection Wait'] = df5['Rejection Date']-df5['Application Date']

df5['Rejection Wait'] = df5['Rejection Wait'].dt.days

# Average rejection wait time (in days) from companies that HAVE responded
avg_rej_wait = int(df5['Rejection Wait'].sum() / len(df5))
df_rej_wait = pd.DataFrame([{'Average wait time (d)':avg_rej_wait}]) 

# Number of applications rejected
napps_rej_sum = df2['Rejection Date'].count()

# Number of first interviews scheduled
ninterviews1 = df2['Interview 1 Date'].count()

# Number of second interviews scheduled
ninterviews2 = df2['Interview 2 Date'].count()

# Percentage of applications that have NOT responded at all
perc_no_resp = int((df3['Submissions'].sum() 
                 - df4['Rejections'].sum() 
                 - ninterviews1 - ninterviews2) 
                / df3['Submissions'].sum() 
                * 100)

# Percentage of applications that HAVE responded
perc_resp = int((1-(df3['Submissions'].sum() 
                       - df4['Rejections'].sum() 
                       - ninterviews1 - ninterviews2) 
                    / df3['Submissions'].sum()) 
                   * 100)

# Find unique company names and count (could also use .nunique() to find count...)
unique_co_count = df2['Company Name'].nunique()

# Quick check to show the various methods are equivalent
if np.count_nonzero(
    df2['Company Name'].unique()) == df2['Company Name'].nunique() == unique_co_count: 
    print('True: Pass')
else:
    print('False: Fail')

# Days between application date and today for companies that haven't responded

mask4 = df2[['Application Date','Interview 1 Date', 'Rejection Date']].isnull()
df6 = df2[mask4['Interview 1 Date'] & mask4['Rejection Date']]
df6.insert(df.shape[1], 'Response Wait (d)', dt.datetime.today())

mask5 = df6['Response Wait (d)'].notnull()
df7 = df6[mask5][['Application Date', 'Response Wait (d)']]

df7['Response Wait (d)'] = df7['Response Wait (d)']-df7['Application Date']

df7['Response Wait (d)'] = df7['Response Wait (d)'].dt.days

df8 = pd.merge(
    df2,df7, left_index=True, right_index=True, 
    how='left', indicator=True).sort_index()

# Number of Applications that have gone unresponed to
nno_resp = df8['Response Wait (d)'].count() 

# Number of applications that have received some response
nyes_resp = df8['Application Date_x'].count() - df8['Response Wait (d)'].count()

# Number of Application submitted validation check
# napps_sum == df8['Application Date_x'].count()

# Total number of apps should add up to (responed to + not responed to) validation check
# nno_resp+nyes_resp == napps_sum

# Extract the sum of days waiting by date
resp_wait_tot = df8.groupby('Application Date_x', 
                            as_index=False)['Response Wait (d)'].sum()
# Raname columns
resp_wait_tot.columns = ['Application Date_x', 'Avg days waiting']

# New dataframe with columns grouped by date
df9 = pd.DataFrame(resp_wait_tot)

# Create cumulitive sums
resp_wait_cum = pd.DataFrame(np.cumsum(df9['Avg days waiting']))


# Add cumulitive sums to dataframe
resp_wait_cum.columns = ['Cumulative days waiting']

df10=pd.concat([df9, resp_wait_cum], axis = 1)

avg_days_wait = int(df10['Avg days waiting'].sum() / nno_resp)


#-------------

# External styling sheet (CSS)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Dashboard Title
header = html.H2(children="Roo's Interactive Job Opportunity Search Tracker")
sub_header = html.H6(children="(Automatically updates daily at 5:01pm Mountain Time)")

#--------------

#Create charts

# Application Summary Chart

fig9 = make_subplots(y_title='Count<br><sup><sup>(Y-scale hidden for privacy)<sup></sup>', x_title='Date', 
                     rows=2, cols=1, shared_xaxes=True, vertical_spacing = .04)

fig9.update_annotations(font_size=24, 
                        font=dict(family='Montserat, monospace', color="RebeccaPurple"))

fig9.add_trace(go.Scatter( 
              x=df3['Application Date'],
              y=df3['Submissions'], name='Submissions',
     mode="lines", line=dict(color='#42646e')), row=2, col=1)

# fig9.add_trace(go.Scatter( 
#               x=df3['Application Date'],
#               y=df3['Rejections'], name='Rejections', 
#      mode="lines", line=dict(color='#ccb15c')), row=2, col=1)

fig9.add_trace(go.Scatter( 
              x=df3['Application Date'],
              y=df4['Cumulative Submissions'], name='Cumulative Submissions', 
     mode="lines", line=dict(color='#7472d6')), row=1, col=1)

# fig9.add_trace(go.Scatter( 
#               x=df3['Application Date'],
#               y=df4['Cumulative Rejections'], name='Cumulative Rejections', 
#      mode="lines", line=dict(color='#cec0ae')), row=1, col=1)


fig9.update_layout(title='Application Summary <br><sup><sup>Daily and cumulative application submissions<sup></sup>', 
                   legend_title='',
                  #xaxis_title='Date', 
                  #yaxis_title='Count',
                   plot_bgcolor='white',
                   hovermode=False,
                   font=dict(family='Montserat, monospace',
                            size=18, color="RebeccaPurple",
                           ),
                   legend=dict(orientation="v", itemwidth=30, #title='Click Me',
                              bgcolor="#FFFFFF", bordercolor="Black", borderwidth=.5, 
                              yanchor="top", y=.99, 
                              xanchor="left", x=.01,
                              font=dict(family="Courier", size=12, color="black"),
                             )
                  )

fig9.update_xaxes(showline=True, linecolor='black')
fig9.update_yaxes(showline=True, linecolor='black', gridcolor='lightgray', showticklabels=False)

# Salary range histogram chart

fig8 = px.histogram(df2,x=["Pay Low","Pay High"],
                    barmode='group', nbins=12, marginal='box',
                   text_auto=False, opacity=1, color_discrete_sequence=['#ecd59f','#7097a8'])

fig8.update_annotations(font_size=24, 
                        font=dict(family='Montserat, monospace', color="RebeccaPurple"))

fig8.update_layout(title='Salary Ranges<br><sup><sup>Count of jobs applied to that posted a low-end to high-end salary range<sup></sup>', 
                   legend_title='', hovermode=False,
                   xaxis_title='Salary in USD (binned)', 
                   yaxis_title='Count<br><sup><sup>(Y-scale hidden for privacy)<sup></sup>',
                   xaxis_tickprefix = '$', xaxis_tickformat = ',.0f',
                   plot_bgcolor='white',
                   font=dict(family='Montserat, monospace',
                            size=18, color="RebeccaPurple",
                           ),
                   legend=dict(orientation="v", itemwidth=30, #title='Click Me',
                              bgcolor="#FFFFFF", bordercolor="Black", borderwidth=.5, 
                              yanchor="top", y=.67, 
                              xanchor="left", x=.82,
                              font=dict(family="Courier", size=12, color="black"),
                             )
                  )

fig8.update_yaxes(visible=True, showticklabels=False)
#fig8.update_xaxes(showline=True, linecolor='black')


# Chart Summary Statistics

figb1 = make_subplots(
    rows=2, cols=2,
    shared_yaxes=False, horizontal_spacing=.15,
    subplot_titles=("Time looking for the<br> next opportunity", 
                    "Avg time for companies to<br>respond to an application", 
                    "Submissions that have not<br> received any response",
                    "Median salaries for the low<br> and high ranges posted")
)

figb1.add_trace(
    go.Bar(y=df_days_looking['Days on the market'], 
           orientation="v",
           marker_color = '#ff8b8b',
           name='Days', #text=df_days_looking['Days on the market'],
           hoverinfo='skip',
          ),
    row=1,
    col=1,
)

figb1.add_trace(
    go.Bar(y=[perc_no_resp], 
           orientation="v",
           marker_color = '#91a3d2', 
           name='%', #text=perc_no_resp,
           hoverinfo='skip',
          ),
    row=2,
    col=1,
)

figb1.add_trace(
    go.Bar(y=[avg_days_wait], 
           orientation="v",
           marker_color = '#6ffd58', 
           name='Days', text=avg_days_wait,
           hoverinfo='skip',
          ),
    row=1,
    col=2,
)

figb1.add_trace(
    go.Bar(y=[low_pay_med, high_pay_med],
           orientation="v", 
           marker_color = ['#856aa1','#e1ff81'], 
           name='$', text=[low_pay_med, high_pay_med],
           hoverinfo='skip',
          ),
    row=2,
    col=2,
)

figb1.update_xaxes(visible=True, showticklabels=False,
                   showline=True, linecolor='black',
                   title_text='', row=1, col=1)
figb1.update_xaxes(visible=True, showticklabels=False,
                   showline=True, linecolor='black',
                   title_text='', row=1, col=2)
figb1.update_xaxes(visible=True, showticklabels=False,
                   showline=True, linecolor='black',
                   title_text='', row=2, col=1)
figb1.update_xaxes(visible=True, showticklabels=False,
                   showline=True, linecolor='black',
                   title_text='', row=2, col=2)

figb1.update_yaxes(title_text='Days<br><sup><sup>(Y-scale hidden for privacy)<sup></sup>',
                   row=1, col=1, range=[0,days_looking*1.5],showticklabels=False)
figb1.update_yaxes(title_text='Days', row=1, col=2, range=[0,avg_days_wait*1.5])
figb1.update_yaxes(title_text='%<br><sup><sup>(Y-scale hidden for privacy)<sup></sup>', 
                   row=2, col=1, range=[0,100], showticklabels=False)
figb1.update_yaxes(title_text='$', row=2, col=2, range=[0,high_pay_med*1.5])


figb1.update_layout(title='Summary Statistics', 
                   #legend_title='df',
                   #xaxis_title='', 
                   #yaxis_title='Days',
                   #xaxis_tickprefix = '', xaxis_tickformat = ',.0f',
                   plot_bgcolor='white',
                   font=dict(family='Montserat, monospace',
                            size=18, color="RebeccaPurple",
                           ), 
                  showlegend=False, width=800, height=500,
                  margin=dict(l=60,r=60,t=120,b=50),
                  paper_bgcolor="white",
                  )





# Duplicate charts for testing
chart1 = fig9
chart2 = fig8
chart3 = figb1
# chart4 = fig9


graph1 = dcc.Graph(
        id='graph1',
        figure=chart1,
        className="six columns"
    )


graph2 = dcc.Graph(
        id='graph2',
        figure=chart2,
        className="six columns"
    )


graph3 = dcc.Graph(
        id='graph3',
        figure=chart3,
        className="six columns"
    )


# graph4 = dcc.Graph(
#         id='graph4',
#         figure=chart4,
#         className="six columns"
#     )



row1 = html.Div(children=[graph1, graph3],)
# row2 = html.Div(children=[graph2, graph4])
row2 = html.Div(children=[graph2])

# Footer
footer = html.H6(children="(Some details hidden for privacy)")

layout = html.Div(children=[header, sub_header, row1, row2, footer], style={"text-align": "center"})

app.layout = layout

if __name__ == "__main__":
    app.run_server(debug=True)
