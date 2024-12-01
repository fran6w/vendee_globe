# coding: utf-8

# imports
import datetime
import glob
import os
import re

import requests

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import ipywidgets as widgets

from bs4 import BeautifulSoup

import plotly.express as px
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)

from catboost import CatBoostRegressor
import shap

# patch pour ajouter le tag xxid à openpyxl
from openpyxl.styles.cell_style import CellStyle

CellStyle.__attrs__ = ("numFmtId", "fontId", "fillId", "borderId",
                       "applyAlignment", "applyProtection", "pivotButton", "quotePrefix", "xfId",
                       "xxid")  # add xxid tag
def new_init(self,
             numFmtId=0,
             fontId=0,
             fillId=0,
             borderId=0,
             xfId=None,
             xxid=None,  # add xxid tag
             quotePrefix=None,
             pivotButton=None,
             applyNumberFormat=None,
             applyFont=None,
             applyFill=None,
             applyBorder=None,
             applyAlignment=None,
             applyProtection=None,
             alignment=None,
             protection=None,
             extLst=None,
            ):
    self.numFmtId = numFmtId
    self.fontId = fontId
    self.fillId = fillId
    self.borderId = borderId
    self.xfId = xfId
    self.xxid = xxid  # add xxid tag
    self.quotePrefix = quotePrefix
    self.pivotButton = pivotButton
    self.applyNumberFormat = applyNumberFormat
    self.applyFont = applyFont
    self.applyFill = applyFill
    self.applyBorder = applyBorder
    self.alignment = alignment
    self.protection = protection

CellStyle.__init__ = new_init


# data collection
def update_data(data_folder, verbose=True):
    """Collect last Excel files from the current race"""

    last_file = sorted(glob.glob(f"{data_folder}/*.xlsx"))[-1]
    last_file = os.path.basename(last_file)
    last_date = last_file.split("_")[1]  # extract date part from "vendeeglobe_YYYYMMDD_HHMMSS.xlsx"
    last_date = datetime.datetime.strptime(last_date, "%Y%m%d")

    # check that we are updating the current race
    if last_date.year < datetime.datetime.today().year - 1:
        dates = []
    else:
        dates = pd.date_range(last_date, datetime.datetime.today(), freq="1D")

    flag = False

    for day in dates:
        for hour in range(2, 24, 4):
            date = f'{day:%Y%m%d}_{hour:02}0000'
            filename = f"{data_folder}/vendeeglobe_{date}.xlsx"

            if os.path.exists(filename):
                continue

            url = f'https://www.vendeeglobe.org/sites/default/files/ranking/vendeeglobe_leaderboard_{date}.xlsx'

            try:
                r = requests.get(url)
                if r.status_code == 200:
                    with open(filename, 'wb') as out_file:
                        out_file.write(r.content)
                        if verbose: print('write', filename)
                    flag = True
                else:
                    break

            except requests.exceptions.HTTPError as exception:
                print(f'error HTTP: {url}')

    if not flag:
        if verbose: print("no new data")


# data load
def load_race_data(data_folder):
    """Build a dataframe from the collected Excel files"""

    files = sorted(glob.glob(f"{data_folder}/*.xlsx"))

    dfs = []
    for file in files:
        df = pd.read_excel(file, skiprows=4, skipfooter=4, na_values=['NL', 'RET'])

        # test arrivals
        test = df.iloc[:, 1].astype(str).str.endswith('ARV').any()
        if test:
            n = df.iloc[:, 1].astype(str).str.endswith('ARV').sum()
            df = pd.read_excel(file, skiprows=n+6, skipfooter=4, na_values=['NL', 'RET'])

        # A column empty
        df = df.drop(df.columns[0], axis=1)

        # renaming columns
        df.columns = ['rang', 'nat_voile', 'skipper_voilier',
                      'heure', 'latitude', 'longitude',
                      'cap_30min', 'vitesse_30min', 'VMG_30min', 'distance_30min',
                      'cap_last', 'vitesse_last', 'VMG_last', 'distance_last',
                      'cap_24h', 'vitesse_24h', 'VMG_24h', 'distance_24h',
                      'DTF', 'DTL']

        # timestamp in file name
        df['date'] = file[-20:-5]

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df = df.loc[df['rang'].notnull()]
    df['rang'] = df['rang'].astype(int)

    return df


# data prep
def get_skippers(df, start, stop):
    """Retrieve all skippers"""

    skippers = (df[['date', 'skipper', 'DTF']]
                .sort_values('date')
                .drop_duplicates(subset='skipper', keep='last')
                .sort_values('DTF')
                .iloc[start:stop]
                .loc[:, 'skipper']
                .values
               )

    return skippers


def data_prep_race(df, skipper_corrections=None):
    """"Data prep of race data"""

    if skipper_corrections is None:
        skipper_corrections = []

    # transform numeric columns
    for col in ['cap_30min', 'vitesse_30min', 'VMG_30min', 'distance_30min',
                'cap_last', 'vitesse_last', 'VMG_last', 'distance_last',
                'cap_24h', 'vitesse_24h', 'VMG_24h', 'distance_24h',
                'DTF', 'DTL']:
        df[col] = df[col].str.extract('([\d\.]+)', expand=False).astype(float)

    # dates
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d_%H%M%S')

    # nat_voile
    tab = df['nat_voile'].str.extract('([A-Z]{3})\s*(\d+)')
    df['nat_voile'] = tab[0] + tab[1]

    # skipper_bateau
    df['skipper_voilier'] = df['skipper_voilier'].str.title()
    df[['skipper', 'voilier']] = df['skipper_voilier'].str.split('\n', expand=True)
    for skipper1, skipper2 in skipper_corrections:
        df['skipper'] = df['skipper'].str.strip().replace(skipper1, skipper2)

    # latitude
    tab = df['latitude'].str.extract("(\d+)°([\d\.]+)'([NS])")  # degrees, minutes, N or S
    df['latitude'] = (tab[0].astype(float) + tab[1].astype(float)/60) * tab[2].map({'N': 1, 'S': -1})

    # longitude
    tab = df['longitude'].str.extract("(\d+)°([\d\.]+)'([EW])")  # degrees, minutes, E or W
    df['longitude'] = (tab[0].astype(float) + tab[1].astype(float)/60) * tab[2].map({'E': 1, 'W': -1})

    # colors for graphics according to the current rank
    skippers = get_skippers(df, 0, df['skipper'].nunique())
    colors = (px.colors.qualitative.Plotly * 4)[:len(skippers)]
    mapping = dict(zip(skippers, colors))
    df['color'] = df['skipper'].map(mapping)

    return df


# Web data collection and preparation

def clean_text(text, title=True):
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    if title:
        text = text.title()
    return text


def load_web_data_2020():

    # chargement de la page glossaire
    html = open("Glossaire - Vendée Globe.htm").read()
    soup = BeautifulSoup(html, features="lxml")

    rows = []

    # recherche des tags h3, class="boats-list__skipper-name"
    # on crée un dictionnaire par voilier
    for span_tag in soup.findAll('h3', {'class': 'boats-list__boat-name'}):
        rows.append({'boat-name': clean_text(span_tag.text)})

    # recherche des tags span, class="boats-list__skipper-name"
    for i, span_tag in enumerate(soup.findAll('span', {'class': 'boats-list__skipper-name'})):
        row = rows[i]

        # pour faciliter la comparaison ensuite on effectue un peu de nettoyage ici
        row['skipper-name'] = clean_text(span_tag.text)

    # recherche des tags h3, class='boats-list__popup-title'
    for i, h3_tag in enumerate(soup.findAll('h3', {'class': 'boats-list__popup-title'})):
        row = rows[i]

        # recherche des sous-tags li
        for li_tag in h3_tag.parent.findAll('li'):
            key, value = li_tag.text.split(':', 1)
            row[key.strip()] = clean_text(value, False)

    df = pd.DataFrame(rows)
    df['foil'] = df['Nombre de dérives'].str.contains('foil').astype(int)
    df = df.rename(columns={'skipper-name': 'skipper'})

    return df


def load_web_data_2024():
    """Web scrapping of the Vendee Globe web site"""

    r = requests.get("https://www.vendeeglobe.org/skippers")
    soup = BeautifulSoup(r.content)

    rows = []

    for tag in soup.find_all("a", {"class": "outline-none"}):
        row = {}
        for first_name_tag in tag.find_all("span", {"class": "skipper-info__first-name"}):
            row["first_name"] = first_name_tag.text
            break
        for last_name_tag in tag.find_all("span", {"class": "skipper-info__last-name"}):
            row["last_name"] = last_name_tag.text
            break
        for team_tag in tag.find_all("span", {"class": "skipper-info__team"}):
            row["team"] = team_tag.text
            break

        r2 = requests.get(f"https://www.vendeeglobe.org{tag.attrs['href']}")
        soup2 = BeautifulSoup(r2.content)

        for tag2 in soup2.find_all("div", {"class": "text text--light"}):
            for li_tag in tag2.find_all("li"):
                key, value = li_tag.text.split(':', 1)
                row[key.strip()] = clean_text(value, False)

        rows.append(row)

    return pd.DataFrame(rows)

def data_prep_web(df, skipper_corrections=None):
    """Data preparation of web data"""

    if skipper_corrections is None:
        skipper_corrections = []

    if "skipper" not in df:
        df["skipper"] = df["first_name"].str.strip() + " " + df["last_name"].str.strip()
    for skipper1, skipper2 in skipper_corrections:
        df['skipper'] = df['skipper'].replace(skipper1, skipper2)

    # processing of numeric columns
    for col in ['Longueur', 'Largeur', 'Tirant d\'eau', 'Déplacement (poids)',
                'Hauteur mât', 'Surface de voiles au près', 'Surface de voiles au portant', 'Poids']:

        if col in df:
            df[col] = df[col].fillna('')
            df[col] = df[col].str.extract('([\d,]+)')
            df[col] = df[col].str.replace(',', '.')
            df[col] = df[col].replace('', np.nan)
            df[col] = df[col].replace('NC', np.nan)
            df[col] = df[col].astype(float)

    # cleaning
    df["Architecte"] = (df["Architecte"].
                        str.strip()
                        .str.replace("[/–]", "-", regex=True)
                        .str.replace(" ?- ?", " - ", regex=True)
                        )

    return df

# Wikipedia data collection and preparation

def load_wiki_data_2024():
    """Collection of of Wikipedia page"""
    var = pd.read_html("https://fr.wikipedia.org/wiki/Vend%C3%A9e_Globe_2024-2025")
    df_wiki = var[2]
    df_wiki = df_wiki.set_axis(["genre", "skipper", "nationalité", "age", "participations",
                                "bateau", "appendices", "architecte", "chantier", "annee"],
                               axis=1)

    for col in ["age", "participations"]:
        df_wiki[col] = df_wiki[col].str.extract('([\d\.]+)').astype(float)

    df_wiki["foil"] = df_wiki["appendices"].map({"foils": 1, "dérives droites": 0})
    df_wiki["genre"]= df_wiki["genre"].map({"♂": 1, "♀": 0})
    df_wiki["annee"] = df_wiki["annee"].astype(int)
    df_wiki["architecte"] = df_wiki["architecte"].str.strip().str.replace("[/–]", "-", regex=True)
    df_wiki["architecte"] = df_wiki["architecte"].str.replace(" ?- ?", " - ", regex=True)

    return df_wiki

# Compute impact of foil
def impact_foil(df):
    tab = df[['foil', 'rang', 'VMG_24h']].copy()
    tab['foil'] = tab['foil'].map({0: 'sans', 1: 'avec'})
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle('Impact du foil...')
    ax1 = fig.add_subplot(121)
    ax1.set_title('... sur le classement')
    sns.boxenplot(x='foil', y='rang', hue='foil', data=tab, legend=False, ax=ax1)
    ax1.yaxis.set_inverted(True)
    ax2 = fig.add_subplot(122)
    ax2.set_title('... sur la vitesse utile')
    sns.boxenplot(x='foil', y='VMG_24h', hue='foil', data=tab, legend=False, ax=ax2)


# compute foil impact on numeric columns
def impact_foil_on_column(col, aggfunc, scale, df):

    # cas où col = scale = DTF => pas de graphique
    if col == scale == 'DTF':
        print('pas de graphique')
        return None

    # selection et tri
    tab = df[[scale, 'foil', col]].copy().sort_values(scale)
    tab['foil'] = tab['foil'].map({0:'sans', 1:'avec'})

    # DTF arrondies aux centaines pour calculer les moyennes
    if scale == 'DTF':
        tab[scale] = tab[scale].apply(lambda x: round(x, -2))

    # 1. groupby
    tab = tab.groupby([scale, 'foil'])
    # 2. aggfunc
    if aggfunc == 'count':
        tab = tab.count()
    elif aggfunc == 'mean':
        tab = tab.mean()
    elif aggfunc == 'std':
        tab = tab.std()
    elif aggfunc == 'min':
        tab = tab.min()
    elif aggfunc == '25%':
        tab = tab.quantile(.25)
    elif aggfunc == '50%':
        tab = tab.median()
    elif aggfunc == '75%':
        tab = tab.quantile(.75)
    elif aggfunc == 'max':
        tab = tab.max()
    else:
        raise (f'Unknown aggfunc: {aggfunc}')
    # 3. reshape
    tab = tab.unstack().droplevel(0, axis=1)

    # plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f'Impact du foil sur la colonne "{col}" ({aggfunc})')
    ax.set_xlabel(scale)
    ax.set_ylabel(f'{col} ({aggfunc})')

    dates = tab.index
    flag = tab['avec'] >= tab['sans']

    ax.plot(dates, tab['avec'], label='avec foil')
    ax.plot(dates, tab['sans'], label='sans foil')
    ax.fill_between(dates, tab['avec'], tab['sans'], flag, alpha=0.5)
    ax.fill_between(dates, tab['sans'], tab['avec'], ~flag, alpha=0.5)
    ax.legend()

    # axe des x
    if scale == 'date':
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    else:
        ax.invert_xaxis()


# interaction
def interact(df):
    # colonnes numériques avec variance
    num_cols = [col for col in df.columns if is_numeric_dtype(df[col]) and not np.allclose(df.groupby('skipper')[col].var().dropna(), 0)]

    # dropdown columns
    column = widgets.Dropdown(options=num_cols,
                              value='VMG_24h',
                              description='Colonne :',
                              )
    # dropdown aggfunc
    aggfunc = widgets.Dropdown(options=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
                               value='mean',
                               description='Agrégation :'
                               )

    # radio button scale
    scale = widgets.RadioButtons(options=['date', 'DTF'],
                                 value='date',
                                 description='Echelle :'
                                 )

    widgets.interact(impact_foil_on_column, col=column, aggfunc=aggfunc, scale=scale, df=widgets.fixed(df))


# Display ranking of top n skipper
def show_ranking(df, n=10):
    """Display ranking of top n skippers"""

    dates = sorted(df['date'].unique())[-12:]

    skippers = (df[['date', 'skipper', 'DTF', 'foil']]
                .sort_values('date')
                .drop_duplicates(subset='skipper', keep='last')
                .sort_values('DTF')
                .iloc[0:n]
               )

    fig, ax = plt.subplots(figsize=(10, 5))
    tab = (df.loc[df.skipper.isin(skippers.skipper.values) & df.date.isin(dates), ["skipper", "date", "rang"]]
           .astype({"skipper": pd.CategoricalDtype(categories=skippers.skipper.values, ordered=True)})
           .sort_values(["skipper", "date"])
           .assign(date=lambda df_: df_.date.dt.strftime("%d/%m\n%Hh"))
          )
    sns.lineplot(data=tab, x="date", y="rang", hue="skipper", palette=px.colors.qualitative.Plotly, ax=ax)
    ys = np.where(skippers.foil == 0.0)[0]
    ax.scatter([ax.get_xlim()[1]-0.4] * len(ys), ys+1.0, marker='v', color=cm.tab10.colors[1], s=10)
    ys = np.where(skippers.foil == 1.0)[0]
    ax.scatter([ax.get_xlim()[1]-0.5] * len(ys), ys+1.0, marker='^', color=cm.tab10.colors[0], s=10)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_inverted(True)
    ax.legend(bbox_to_anchor=(1.25, 1.0))
    ax.set_title("Classements des 10 premiers au cours des 2 derniers jours")


# Visualisation de la distance parcourue par voilier et par jour
def show_race(df):
    # pretty date
    def pretty_date(dt, timeframe):

        # print month
        if (dt.day == 1) or (dt == timeframe[0]):
            return dt.strftime('%b')

        # print day
        elif ((dt.day % 5) == 0) or (dt == timeframe[-1]):
            return dt.strftime('%d')

        # pass
        return ''

    # calcul de la distance parcourue (DTF max - DTF courant)
    tab = df.groupby(['skipper', 'date'])['DTF'].min().unstack('skipper')
    _max = tab.max().max()
    tab = tab.fillna(_max)
    # on prend le max par jour
    tab = tab.resample('D').min()
    tab = tab.reset_index(drop=True)
    tab.loc[0] = _max
    # on trie selon la distance actuelle
    tab = tab.sort_values(tab.index[-1], ascending=False, axis=1)

    # figure
    fig, ax = plt.subplots(figsize=(18,10))
    fig.suptitle('Vendée Globe 2024-2025')
    ax.set_title('Distance quotidienne parcourue par les skippers')
    colors = df.set_index('skipper')['color'].to_dict()
    for i, (skipper, ser) in enumerate(tab.items()):
        # cas standard le voilier est à sa distance parcourue max
        if ser.min() == ser.iloc[-1]:
            ax.plot(ser, [i] * len(ser), marker='^', color=colors[skipper])
        # le voilier n'est pas à sa distance parcourue max
        # on affiche en pointillés sa distance parcourue max
        # on affiche en plein où il en est
        else:
            n = ser.idxmin()
            ax.plot(ser, [i] * len(ser), marker='^', lw=0, color=colors[skipper])
            ax.plot(ser.iloc[[0, -1]], [i, i], color=colors[skipper])

    # print dates
    timeframe = df.set_index('date').resample('D').size().index
    days = [pretty_date(dt, timeframe) for dt in timeframe]
    for i, x in enumerate(tab.iloc[:,-1]):
        ax.annotate(days[i], (x, tab.shape[1]-0.5), fontsize=9)

    # labels des x
    ax.set_xlabel('DTF (Distance To Finish) en milles marins')
    ax.invert_xaxis()
    # labels des y
    ax.set_yticks(range(len(tab.columns)))
    ax.set_yticklabels(tab.columns)

# map of skippers
def show_globe(df, start, stop, projection='orthographic'):
    """Display a map with the skippers ranked from start to stop"""

    dt = df['date'].max().strftime('%d/%m/%Y %Hh')

    skippers = get_skippers(df, start, stop)

    df2 = (df
           .loc[lambda df_: df_.skipper.isin(skippers)]
           .sort_values(['DTF', 'date'])
           .astype({"rang": str})
           )

    last_skipers = df2.groupby('skipper')['date'].max()
    ret_skippers = last_skipers.loc[last_skipers != df2['date'].max()].index
    df2.loc[df2.skipper.isin(ret_skippers), "rang"] = "abandon"

    fig = px.line_geo(df2, lat='latitude', lon='longitude', hover_name="skipper",
                        hover_data={'skipper': False, 'voilier': True, 'rang': True},
                        color='skipper', projection=projection)

    fig.update_layout(showlegend = True,
                        height=500,
                        title_text = f'Vendée Globe au {dt}<br />rang {start+1} - {stop}',
                        geo = dict(
                            showland = True,
                            showcountries = True,
                            showocean = True,
                            countrywidth = 0.5,
                            landcolor = 'tan',
                            lakecolor = 'aliceblue',
                            oceancolor = 'aliceblue',
                            lonaxis = dict(
                                showgrid = True,
                                gridcolor = 'rgb(102, 102, 102)',
                                gridwidth = 0.5),
                            lataxis = dict(
                                showgrid = True,
                                gridcolor = 'rgb(102, 102, 102)',
                                gridwidth = 0.5)))

    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )

    return fig


# Run SHAP model to collect the effects of numeric features
def show_shap_values(df):
    """Show SHAP beeswarm plot on numeric features to predict the rank"""

    X = df.select_dtypes('number')
    y = -X["rang"]  # opposite of rank to have a positive effect directed to the right and negative effects to the left
    X = X.drop(["rang", "DTL", "DTF"], axis=1)

    model = CatBoostRegressor(iterations=300, learning_rate=0.1, random_seed=123)
    model.fit(X, y, verbose=False, plot=False)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # summarize the effects of all the features
    shap.plots.beeswarm(shap_values, max_display=11)


# Display the average speed by degrees of latitude and longitude
def show_speed(df):
    """Display the average speed by degrees of latitude and longitude"""

    # Display of positions
    def longlat2id(long, lat):
        we = 'W' if long < 0 else 'E'
        sn = 'S' if lat < 0 else 'N'
        return f"{abs(lat)}° {sn}, {abs(long)}° {we}"

    # Build a geojson data structure adapted to the current race for choropleth graph
    def make_feature(long, lat):
        d = dict(type="Feature",
                 properties=dict(position=longlat2id(long, lat)),
                 geometry=dict(type="Polygon", coordinates=[[[long,lat],[long+1,lat],[long+1,lat+1],[long,lat+1],[long,lat]]]))
        return d

    # Compute the average speed by degrees of latitude and longitude
    tab = df[['latitude', 'longitude', 'vitesse_last']].copy()
    tab[['latitude', 'longitude']] = np.floor(tab[['latitude', 'longitude']]).astype(int)

    pivot = (tab
             .pivot_table(index='latitude',
                          columns='longitude',
                          values='vitesse_last')
            )

    tab = pivot.stack().dropna().round(1).reset_index().rename(columns={0:"vitesse"})
    tab["position"] = tab["longitude"].combine(tab["latitude"], longlat2id)

    longs_lats = tab[["longitude", "latitude"]].drop_duplicates().values

    geojson = dict(type="FeatureCollection",
                 features=[make_feature(long, lat) for long, lat in longs_lats])

    fig = px.choropleth_mapbox(data_frame=tab,
                                    geojson=geojson,
                                    locations='position',
                                    color="vitesse",
                                    featureidkey='properties.position',
                                    color_continuous_scale="Bluered",
                                    mapbox_style="carto-positron",
                                    height=600,
                                    width=800,
                                    center={'lat':tab["latitude"].mean(), 'lon':tab["longitude"].mean()},
                                    zoom=0.7)
    fig.update_traces(marker_line_width=0)
    fig.update_layout(title=dict(text="Vitesse moyenne des voiliers"))
    return fig
