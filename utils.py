import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import uuid
import operator
import calendar
import time
import os
import sys
import math
import random
import pandas as pd
import numpy as np
import seaborn as sns
from sqlalchemy import create_engine
import pickle
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from turbodbc import connect as con
from turbodbc import make_options
from datetime import date
import datetime
from datetime import timedelta
import multiprocessing as mp
from bokeh.io import show
from bokeh.io import save
from bokeh.io import output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import Range1d
from bokeh.models import HoverTool
from bokeh.models import LabelSet
from bokeh.models import DatetimeTickFormatter
from bokeh.layouts import column
from pylatex import MultiColumn
from scipy import optimize
from scipy.stats import beta
from scipy.stats import truncnorm
from tabulate import tabulate
pd.options.display.float_format = '{:,.2f}'.format
from matplotlib import pyplot as plt
from  matplotlib.ticker import PercentFormatter
import matplotlib.ticker as ticker
sns.set_style('whitegrid')
sns.set_palette('husl')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100



class _dotdict(dict):
    def __init__(self, *args):
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self.__dict__[k] = v
                    self[k] = v
    
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr)

    def __setattr__(self, k, v):
        self.__dict__[k] = v
        self[k] = v

    def __setitem__(self, k, v):
        super().__setitem__(k, v)

    def __repr__(self):
        try:
            print_df = pd.DataFrame(self.__dict__)
        except:
            print_df = pd.DataFrame(self.__dict__, index=[0]).transpose()
        return pretty_print(print_df)

def pretty_print(df):
    return tabulate(df, tablefmt='grid', floatfmt='.2f')

def create_dir(tree):
    if not os.path.exists(tree.name):
        os.makedirs(tree.name)
        os.makedirs(tree.name + '\\' + 'archive')
    if not os.path.exists('temp'):
        os.makedirs('temp')

def connect(conn_dict):
    options = make_options(prefer_unicode=True)
    if conn_dict['credentials'] is None:
        return con(driver=conn_dict['driver'],
                   server=conn_dict['server'],
                   database=conn_dict['database'],
                   trusted_connection='yes',
                   turbodbc_options=options)
    return con(driver=conn_dict['driver'],
               server=conn_dict['server'],
               database=conn_dict['database'],
               uid=conn_dict['credentials']['user'],
               pwd=conn_dict['credentials']['password'],
               turbodbc_options=options)

def engine(conn_dict):
    if conn_dict['credentials'] is None:
        return create_engine(str('mssql+turbodbc://' + conn_dict['server'] +
                                  '/' + conn_dict['database'] +
                                  '?driver=ODBC+Driver+17+for+SQL+Server'))
    else:
        return create_engine(str('mssql+turbodbc://' +
                                 conn_dict['credentials']['user'] + ':' +
                                 conn_dict['credentials']['password'] +
                                 '@' + conn_dict['server'] + '/' +
                                 conn_dict['database'] +
                                 '?driver=ODBC+Driver+17+for+SQL+Server'))

def load_all_scenarios(tree):
    conn = connect(tree.connection_dict)
    query = str('select * from scenarios')
    return pd.read_sql(query, conn)


def save_scenario_to_sql(tree, scenario, overwrite):
    conn = connect(tree.connection_dict)  
    query = str('select scenario from scenarios')
    scen = pd.read_sql(query, conn)['scenario'].values
    if scenario.scenario in scen:
        if not overwrite:
            print('scenario exists in database, set overwrite to True to replace')
            return
        else:
            cursor = conn.cursor()
            query = str('delete from scenarios '
                        'where scenario = \'' + scenario.scenario + '\'')
            cursor.execute(query)
            conn.commit()
            cursor.close()
    eng = engine(tree.connection_dict)
    try:
        df = pd.DataFrame(scenario)
    except:
        df = pd.DataFrame(scenario, index=[0])
    df.to_sql(name='scenarios', con=eng,
               if_exists='append', method='multi',
               index=False, chunksize=500)
    print('scenario', scenario.scenario, 'added to database')
    print(tree.scenarios[scenario.scenario])
    return

def load_scenario(branch, scenario):
    conn = connect(branch.tree.connection_dict)
    query = str('select * from scenarios ' +
                'where scenario = \'' + scenario + '\'')
    df = pd.read_sql(query, conn)
    if df.empty:
        print('scenario does not exist')
        return
    else:
        return _dotdict(df.to_dict(orient='records')[0])

def check_scenario(branch, scenario):
    conn = connect(branch.tree.connection_dict)
    for k, v in scenario.items():
        if k == 'autoforecaster' and v is not None:
            query = str('select top 10 * from autoforecaster ' +
                        'where scenario = \'' + v + '\'')
            temp = pd.read_sql(query, conn)
            if temp.empty:
                print('no', v, 'scenario for autoforecaster')
                return False

        if k == 'capacity_volumes' and v is not None:
            query = str('select top 10 * from input_capacity_volumes ' +
                        'where scenario = \'' + v + '\'')
            temp = pd.read_sql(query, conn)
            if temp.empty:
                print('no', v, 'scenario for capacity volumes')
                return False

        if k == 'economics' and v is not None:
            query = str('select top 10 * from economics ' +
                        'where scenario = \'' + v + '\'')
            temp = pd.read_sql(query, conn)
            if temp.empty:
                print('no', v, 'scenario for economics')
                return False

        if k == 'price_deck' and v is not None:
                query = str('select top 10 * from pricing ' +
                            'where price_deck = \'' + v + '\'')
                temp = pd.read_sql(query, conn)
                if temp.empty:
                    print('no', v, 'in price decks')
                    return False

        if k == 'properties' and v is not None:
            query = str('select top 10 * from properties ' +
                        'where scenario = \'' + v + '\'')
            temp = pd.read_sql(query, conn)
            if temp.empty:
                print('no', v, 'in properties')
                return False

        if k == 'schedule_inputs' and v is not None:
            query = str('select top 10 * from schedule_inputs ' +
                        'where scenario = \'' + v + '\'')
            temp = pd.read_sql(query, conn)
            if temp.empty:
                print('no', v, 'in schedule inputs')
                return False

        if k == 'forecast' and v is not None:
            query = str('select top 10 * from prod_forecasts ' +
                        'where scenario = \'' + v + '\'')
            temp = pd.read_sql(query, conn)
            if temp.empty:
                print('warning: no', v, 'in forecast\r')

        if k == 'project' and v is not None:
            query = str('select top 10 * from projects ' +
                        'where scenario = \'' + v + '\'')
            temp = pd.read_sql(query, conn)
            if temp.empty:
                print('no', v, 'in projects')
                return False

        if k == 'project_id' and v is not None:
            if v is None and scenario.project is not None:
                print('no ID field provided for project', scenario.project)
                return

        if k == 'area' and v is not None:
            area_list = ', '.join('\'{0}\''.format(a) for a in v.split(', '))
            query = str('select top 10 * from properties ' +
                        'where prospect in (' + area_list + ')')
            temp = pd.read_sql(query, conn)
            if temp.empty:
                print('no', v, 'in areas')
                return False
        if k == 'probability' and v is not None:
            branch.probability = True
    return True

def update_daily_production(tree):
    conn = connect(tree.connection_dict)
    print('inserting new monthly data')
    start = time.time()
    query = str('insert into daily_production '
                'select a.propnum, a.prod_date, '
                'datediff(day, dateadd(day, -1, b.first_prod_date), a.prod_date), '
                'a.gas, a.oil, a.water '
                'from (select p.PROPNUM, d.prod_date, '
                'p.OIL / d.p_days AS oil, '
                'p.GAS / d.p_days AS gas, '
                'p.WATER / d.p_days AS water '
                'from COG_WORKING.dbo.AC_PRODUCT AS p INNER JOIN dbo.datehelper AS d ON p.P_DATE = d.p_date) a '
                'inner join (select distinct propnum, first_prod_date from properties) b '
                 'on a.propnum = b.propnum and a.prod_date >= b.first_prod_date '
                'where not exists (select * from daily_production '
                'where daily_production.idp = a.propnum and daily_production.prod_date = a.prod_date) '
                'and a.prod_date <= eomonth(getdate(), -1)')
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    stop = time.time()
    timer(start, stop)
    print('updating daily data')
    start = time.time()
    query = str('update daily_production '
                'set daily_production.gas = cog_working.dbo.ac_daily.GAS, '
	            'daily_production.oil = cog_working.dbo.ac_daily.OIL, '
	            'daily_production.water = cog_working.dbo.ac_daily.WATER '
                'from daily_production inner join cog_working.dbo.ac_daily '
                'on daily_production.idp = cog_working.dbo.ac_daily.propnum '
                'and daily_production.prod_date = cog_working.dbo.ac_daily.d_date '
                'where daily_production.prod_date > dateadd(day, -365, getdate())')
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    stop = time.time()
    timer(start, stop)
    print('inserting new daily data')
    start = time.time()
    query = str('insert into daily_production '
                'select a.propnum, a.d_date, '
                'datediff(day, dateadd(day, -1, b.first_prod_date), a.d_date), '
                'a.gas, a.oil, a.water '
                'from cog_working.dbo.ac_daily a inner join '
                '(select distinct propnum, first_prod_date from properties) b '
                'on a.propnum = b.propnum and a.d_date >= b.first_prod_date '
                'where not exists (select * from daily_production '
                'where daily_production.idp = a.propnum and daily_production.prod_date = a.d_date)')
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    stop = time.time()
    timer(start, stop)
    return

def load_schedule(branch):
    conn = connect(branch.tree.connection_dict)
    query = str('select * from schedule ' +
                'where scenario = \'' + branch.scenario.schedule + '\'')
    return pd.read_sql(query, conn)

def load_schedule_properties(schedule):
    conn = connect(schedule.branch.tree.connection_dict)
    pad_list = ', '.join('\'{0}\''.format(p) for p in schedule.pad_dict.keys())
    query = str('select * from properties ' +
                'where scenario = \'' + schedule.branch.scenario.properties + '\' '
                'and active = 1 '
                'and pad in (' + pad_list + ')')
    return pd.read_sql(query, conn)

def load_properties(branch, idp=None, pad=None, short_pad=None, grouping=None,
                    area=None, scenario=None, project=None, budget_type='base'):
    conn = connect(branch.tree.connection_dict)
    if idp is not None:
        idp_list = ', '.join('\'{0}\''.format(p) for p in idp)
        query = str('select properties.* from properties ' +
                    'inner join economics on properties.propnum = economics.idp '
                    'inner join forecasts on economics.idp = forecasts.idp '
                    'where properties.scenario = \'' + branch.scenario.properties + '\' '
                    'and economics.scenario = \'' + branch.scenario.economics + '\' '
                    'and forecasts.scenario = \'' + branch.scenario.forecast + '\' '
                    'and properties.active = 1 ' 
                    'and properties.propnum in (' + idp_list + ')')
    if pad is not None:
        if type(pad) == list:
            pad_list = ', '.join('\'{0}\''.format(p) for p in pad)
        else:
            pad_list = '\'' + pad + '\''
        query = str('select properties.* from properties ' +
                    'inner join economics on properties.propnum = economics.idp '
                    'inner join forecasts on economics.idp = forecasts.idp '
                    'where properties.scenario = \'' + branch.scenario.properties + '\' '
                    'and economics.scenario = \'' + branch.scenario.economics + '\' '
                    'and forecasts.scenario = \'' + branch.scenario.forecast + '\' '
                    'and properties.pad in (' + pad_list + ')')
    if short_pad is not None:
        if type(short_pad) == list:
            short_pad_list = ', '.join('\'{0}\''.format(p) for p in short_pad)
        else:
            short_pad_list = '\'' + short_pad + '\''
        query = str('select properties.* from properties ' +
                    'inner join economics on properties.propnum = economics.idp '
                    'inner join forecasts on economics.idp = forecasts.idp '
                    'where properties.scenario = \'' + branch.scenario.properties + '\' '
                    'and economics.scenario = \'' + branch.scenario.economics + '\' '
                    'and forecasts.scenario = \'' + branch.scenario.forecast + '\' '
                    'and properties.short_pad in (' + short_pad_list + ')')
    if area is not None:
        if type(area) == list:
            area_list = ', '.join('\'{0}\''.format(a) for a in area)
        else:
            area_list = '\'' + area + '\''
        query = str('select properties.* from properties ' +
                    'inner join economics on properties.propnum = economics.idp '
                    'inner join forecasts on economics.idp = forecasts.idp '
                    'where properties.scenario = \'' + branch.scenario.properties + '\' '
                    'and economics.scenario = \'' + branch.scenario.economics + '\' '
                    'and forecasts.scenario = \'' + branch.scenario.forecast + '\' '
                    'and properties.active = 1 '
                    'and properties.budget_type = \'' + budget_type + '\' '
                    'and properties.prospect in (' + area_list + ')')
    if scenario is not None:
        query = str('select properties.* from properties ' +
                    'inner join economics on properties.propnum = economics.idp '
                    'inner join forecasts on economics.idp = forecasts.idp '
                    'where properties.scenario = \'' + scenario + '\' '
                    'and economics.scenario = \'' + branch.scenario.economics + '\' '
                    'and forecasts.scenario = \'' + branch.scenario.forecast + '\' '
                    'and properties.budget_type = \'' + budget_type + '\' '
                    'and properties.active = 1 ')
    if grouping is not None:
        if type(grouping) == list:
            grouping_list = ', '.join('\'{0}\''.format(g) for g in grouping)
        else:
            grouping_list = '\'' + grouping + '\''
        query = str('select properties.* from properties ' +
                    'inner join economics on properties.propnum = economics.idp '
                    'inner join forecasts on economics.idp = forecasts.idp '
                    'where properties.scenario = \'' + branch.scenario.properties + '\' '
                    'and economics.scenario = \'' + branch.scenario.economics + '\' '
                    'and forecasts.scenario = \'' + branch.scenario.forecast + '\' '
                    'and properties.grouping in (' + grouping_list + ') '
                    'and properties.active = 1 ')
    if project is not None:
        query = str('select * from projects where scenario = \'' + project[0] + '\'')
        project_properties = pd.read_sql(query, conn)['property'].values
        prop_list = ', '.join('\'{0}\''.format(p) for p in project_properties)
        query = str('select properties.* from properties ' +
                    'inner join economics on properties.propnum = economics.idp '
                    'inner join forecasts on economics.idp = forecasts.idp '
                    'where properties.scenario = \'' + branch.scenario.properties + '\' '
                    'and economics.scenario = \'' + branch.scenario.economics + '\' '
                    'and forecasts.scenario = \'' + branch.scenario.forecast + '\' '
                    'and properties.active = 1 ' 
                    'and properties.' + project[1] + ' in (' + prop_list + ')')
    return pd.read_sql(query, conn)

def load_schedule_inputs(branch):
    conn = connect(branch.tree.connection_dict)
    query = str('select * from schedule_inputs ' +
                'where scenario = \'' + branch.scenario.schedule_inputs+ '\'')
    return pd.read_sql(query, conn)

def calc_drill_dates(schedule):
    schedule.schedule_dates.drill_start_date = pd.to_datetime(schedule.schedule_dates.drill_start_date)
    schedule.schedule_dates.drill_end_date = pd.to_datetime(schedule.schedule_dates.drill_end_date)
    for k, rig in schedule.rig_dict.items():
        for idxp, pad in enumerate(rig.pad_list):
            if pad.rig.rig_name == 'DUC':
                for idxw, well in enumerate(pad.well_list):
                    schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                       'drill_start_date'] = pd.NaT
                    schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                       'drill_end_date'] = pd.NaT
                continue
            if pad.drill_start is None:
                prior_pad = rig.pad_list[idxp - 1]
                pad.drill_start = prior_pad.drill_finish + timedelta(prior_pad.mob_out) + timedelta(pad.mob_in)
            if pad.drill_finish is None:
                for idxw, well in enumerate(pad.well_list):
                    idp = well.idp
                    if schedule.uncertainty is not None:
                        u = schedule.uncertainty.loc[schedule.uncertainty.idp == idp].drill_time.values[0]
                    else:
                        u = 1
                    if schedule.risk is not None:
                        r = schedule.risk.loc[schedule.risk.idp == idp].drill_time.values[0]
                    else:
                        r = 1
                    if idxw == 0:
                        well.drill_start_date = pad.drill_start
                        well.drill_end_date = well.drill_start_date + timedelta(well.drill_time * u * r)
                        well.drill_time = well.drill_time * u * r
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'drill_start_date'] = well.drill_start_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'drill_end_date'] = well.drill_end_date
                    elif idxw == len(pad.well_list) - 1:
                        prior_well = pad.well_list[idxw-1]
                        well.drill_start_date = prior_well.drill_end_date
                        well.drill_end_date = well.drill_start_date + timedelta(well.drill_time * u * r)
                        well.drill_time = well.drill_time * u * r
                        pad.drill_finish = well.drill_end_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'drill_start_date'] = well.drill_start_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'drill_end_date'] = well.drill_end_date
                    else:
                        prior_well = pad.well_list[idxw - 1]
                        well.drill_start_date = prior_well.drill_end_date
                        well.drill_end_date = well.drill_start_date + timedelta(well.drill_time * u * r)
                        well.drill_time = well.drill_time * u * r
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'drill_start_date'] = well.drill_start_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'drill_end_date'] = well.drill_end_date
            else:
                drill_time = (pad.drill_finish - pad.drill_start).days / pad.num_wells
                for idxw, well in enumerate(pad.well_list):
                    idp = well.idp
                    if schedule.uncertainty is not None:
                        u = schedule.uncertainty.loc[schedule.uncertainty.idp == idp].drill_time.values[0]
                    else:
                        u = 1
                    if schedule.risk is not None:
                        r = schedule.risk.loc[schedule.risk.idp == idp].drill_time.values[0]
                    else:
                        r = 1
                    if idxw == 0:
                        # if idxp == 0:
                        #     next_pad = rig.pad_list[idxp + 1]
                        #     if next_pad.drill_start is not None:
                        #         pad_timing_delta = (next_pad.drill_start - pad.drill_finish).days
                        if idxp > 0:
                            prior_pad = rig.pad_list[idxp - 1]
                            pad.drill_start = prior_pad.drill_finish + timedelta(pad_timing_delta)
                            if idxp < len(rig.pad_list) - 1:
                                next_pad = rig.pad_list[idxp + 1]
                                if next_pad.drill_start is not None:
                                    pad_timing_delta = (next_pad.drill_start - pad.drill_finish).days
                        well.drill_start_date = pad.drill_start
                        well.drill_end_date = well.drill_start_date + timedelta(drill_time * u * r)
                        well.drill_time = drill_time * u * r
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'drill_start_date'] = well.drill_start_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'drill_end_date'] = well.drill_end_date
                    else:
                        prior_well = pad.well_list[idxw - 1]
                        well.drill_start_date = prior_well.drill_end_date
                        well.drill_end_date = well.drill_start_date + timedelta(drill_time * u * r)
                        well.drill_time = drill_time * u * r
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'drill_start_date'] = well.drill_start_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'drill_end_date'] = well.drill_end_date
                    if idxw == len(pad.well_list) - 1:
                        time_shift = (well.drill_end_date - pad.drill_finish).days
                        pad.time_shift = time_shift
                        pad.drill_finish = well.drill_end_date

def calc_compl_dates(schedule):
    schedule.schedule_dates.compl_start_date = pd.to_datetime(schedule.schedule_dates.compl_start_date)
    schedule.schedule_dates.compl_end_date = pd.to_datetime(schedule.schedule_dates.compl_end_date)
    for k, rig in schedule.rig_dict.items():
        for idxp, pad in enumerate(rig.pad_list):
            if pad.compl_start is None:
                pad.compl_start = (pad.drill_finish + timedelta(pad.mob_out)
                                   + timedelta(pad.log_pad) + timedelta(pad.build_facilities) + timedelta(pad.time_shift))
                prior_pad = rig.pad_list[idxp - 1]
                last_compl_date = prior_pad.compl_finish
                if last_compl_date:
                    if pad.compl_start < last_compl_date:
                        pad.compl_start = last_compl_date + timedelta(3)
            if pad.compl_finish is None:
                for idxw, well in enumerate(pad.well_list):
                    if idxw == 0:
                        well.compl_start_date = pad.compl_start + timedelta(pad.time_shift)
                        well.compl_end_date = well.compl_start_date + timedelta(well.compl_time)
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'compl_start_date'] = well.compl_start_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'compl_end_date'] = well.compl_end_date
                    elif idxw == len(pad.well_list) - 1:
                        prior_well = pad.well_list[idxw-1]
                        well.compl_start_date = prior_well.compl_end_date
                        well.compl_end_date = well.compl_start_date + timedelta(well.compl_time)
                        pad.compl_finish = well.compl_end_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'compl_start_date'] = well.compl_start_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'compl_end_date'] = well.compl_end_date
                    else:
                        prior_well = pad.well_list[idxw - 1]
                        well.compl_start_date = prior_well.compl_end_date
                        well.compl_end_date = well.compl_start_date + timedelta(well.compl_time)
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'compl_start_date'] = well.compl_start_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'compl_end_date'] = well.compl_end_date
            else:
                pad.compl_start = pad.compl_start + timedelta(pad.time_shift)
                pad.compl_finish = pad.compl_finish + timedelta(pad.time_shift)
                compl_time = (pad.compl_finish - pad.compl_start).days / pad.num_wells
                for idxw, well in enumerate(pad.well_list):
                    if idxw == 0:
                        well.compl_start_date = pad.compl_start
                        well.compl_end_date = well.compl_start_date + timedelta(compl_time)
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'compl_start_date'] = well.compl_start_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'compl_end_date'] = well.compl_end_date
                    else:
                        prior_well = pad.well_list[idxw - 1]
                        well.compl_start_date = prior_well.compl_end_date
                        well.compl_end_date = well.compl_start_date + timedelta(compl_time)
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'compl_start_date'] = well.compl_start_date
                        schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                           'compl_end_date'] = well.compl_end_date

def calc_start_dates(schedule):
    schedule.schedule_dates.prod_start_date = pd.to_datetime(schedule.schedule_dates.prod_start_date)
    for k, rig in schedule.rig_dict.items():
        for idxp, pad in enumerate(rig.pad_list):
            if pad.prod_start is None:
                wells = np.arange(0, len(pad.well_list))
                if pad.pod_size == len(wells):
                    pods = [len(wells) - 1]
                elif pad.pod_size == 1:
                    pods = wells
                else:
                    if pad.pod_size > len(wells):
                        pods = [len(wells) - 1]
                    else:
                        pods = [w - 1 for w in wells if (w % pad.pod_size == 0) and (w > 0)]
                        if max(wells) > max(pods):
                            pods.append(max(wells))
                pod_start_dates = [pad.well_list[w].compl_end_date +
                                   timedelta(pad.well_list[w].flowback_time) for w in pods]
                pod_idx = 0
                pod = pods[pod_idx]
                pad.prod_start = pod_start_dates[0]
                pad.prod_finish = pod_start_dates[-1]
                if pad.prod_finish == pad.prod_start:
                    pad.prod_finish = pad.prod_finish + timedelta(3)
                for idxw, well in enumerate(pad.well_list):
                    if idxw > pod:
                        pod_idx += 1
                    pod = pods[pod_idx]
                    well.prod_start_date = pod_start_dates[pod_idx]
                    schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                                'prod_start_date'] = well.prod_start_date
                    if idxw == 0:
                       pad.prod_start = pd.Timestamp(well.prod_start_date)
            else:
                if pad.prod_finish is None:
                    pad.prod_finish = pad.prod_start + timedelta(3)
                for idxw, well in enumerate(pad.well_list):
                    well.prod_start_date = pad.prod_start
                    schedule.schedule_dates.loc[schedule.schedule_dates.idp == well.idp,
                                                'prod_start_date'] = well.prod_start_date
                    if idxw == 0:
                        pad.prod_start = pd.Timestamp(well.prod_start_date)

def save_schedule(schedule):
    conn = connect(schedule.branch.tree.connection_dict)
    eng = engine(schedule.branch.tree.connection_dict)
    cursor = conn.cursor()
    query = str('delete from schedule '
                'where scenario = \'' + schedule.name + '\'')
    cursor.execute(query)
    conn.commit()
    cursor.close()
    schedule.schedule_dates.to_sql(name='schedule', con=eng,
                                  if_exists='append', method='multi',
                                  index=False, chunksize=500)

def save_pad_schedule(schedule):
    eng = engine(schedule.branch.tree.connection_dict)
    # cursor = conn.cursor()
    # query = str('delete from schedule_pads '
    #             'where scenario = \'' + schedule.name + '\'')
    # cursor.execute(query)
    # conn.commit()
    # cursor.close()
    for k, rig in schedule.rig_dict.items():
        for idxp, pad in enumerate(rig.pad_list):
            schedule.schedule_df.loc[schedule.schedule_df['pad'] == pad.pad_name, 'drill_start_date'] = pad.drill_start
            schedule.schedule_df.loc[schedule.schedule_df['pad'] == pad.pad_name, 'drill_end_date'] = pad.drill_finish
            schedule.schedule_df.loc[schedule.schedule_df['pad'] == pad.pad_name, 'compl_start_date'] = pad.compl_start
            schedule.schedule_df.loc[schedule.schedule_df['pad'] == pad.pad_name, 'compl_end_date'] = pad.compl_finish
            schedule.schedule_df.loc[schedule.schedule_df['pad'] == pad.pad_name, 'prod_start_date'] = pad.prod_start
    schedule.schedule_df['run_date'] = pd.Timestamp(schedule.branch.tree.run_time)
    schedule.schedule_df['scenario'] = schedule.name
    schedule.schedule_df['simulation'] = schedule.simulation
    schedule.schedule_df.to_sql(name='schedule_pads', con=eng,
                                if_exists='append', method='multi',
                                index=False, chunksize=500)

def create_gantt_df(schedule):
    temp_df = pd.DataFrame()
    rigs = []
    pads = []
    start = []
    start_text = []
    end = []
    end_text = []
    colors = ['#d3d3d3', '#1b9e77', '#d95f02', '#7570b3']
    color = []
    drill_time = []
    compl_time = []
    well_count = []
    for rig in schedule.rig_dict.values():
        for pad in rig.pad_list:
            avg_drill = round(np.mean(
                [well.drill_time for well in pad.well_list]), 1)
            avg_compl = round(np.mean(
                [well.compl_time for well in pad.well_list]), 1)
            if rig.rig_name != 'DUC':
                rigs.append(rig.rig_name+' - Drill')
                pads.append(pad.pad_name)
                start.append(pad.drill_start)
                start_text.append(pad.drill_start.strftime('%m/%d/%Y'))
                end.append(pad.drill_finish)
                end_text.append(pad.drill_finish.strftime('%m/%d/%Y'))
                color.append(colors[1])
                drill_time.append(avg_drill)
                compl_time.append(avg_compl)
                well_count.append(pad.num_wells)
            rigs.append(rig.rig_name+' - Complete')
            pads.append(pad.pad_name)
            start.append(pad.compl_start)
            start_text.append(pad.compl_start.strftime('%m/%d/%Y'))
            end.append(pad.compl_finish)
            end_text.append(pad.compl_finish.strftime('%m/%d/%Y'))
            color.append(colors[2])
            drill_time.append(avg_drill)
            compl_time.append(avg_compl)
            well_count.append(pad.num_wells)
            rigs.append(rig.rig_name+' - Sales')
            pads.append(pad.pad_name)
            start.append(pad.prod_start)
            start_text.append(pad.prod_start.strftime('%m/%d/%Y'))
            end.append(pad.prod_finish)
            end_text.append(pad.prod_finish.strftime('%m/%d/%Y'))
            color.append(colors[3])
            drill_time.append(avg_drill)
            compl_time.append(avg_compl)
            well_count.append(pad.num_wells)
    temp_df['rig'] = pd.Series(rigs)
    temp_df['pad'] = pd.Series(pads)
    temp_df['start'] = pd.Series(start)
    temp_df['start_text'] = pd.Series(start_text)
    temp_df['end'] = pd.Series(end)
    temp_df['end_text'] = pd.Series(end_text)
    temp_df['color'] = pd.Series(color)
    temp_df['drill'] = pd.Series(drill_time)
    temp_df['compl'] = pd.Series(compl_time)
    temp_df['well_count'] = pd.Series(well_count)
    schedule.gantt_df = temp_df

def create_gantt_chart(schedule, file_path):
    if file_path == 'archive':
        output_file(str(schedule.branch.tree.name + '\\archive\\' +
                        schedule.name + '_' +
                        schedule.branch.tree.run_time + '.html'))
    if file_path == 'main':
        output_file(str(schedule.branch.tree.name + '\\' +
                        schedule.name + '.html'))    
    height = 0.9
    plot_height = 300
    text_font_size = '7pt'
    text_font_style = 'bold'
    angle = 90
    angle_units = 'deg'
    x_offset = 10
    y_offset = 30
    fill_alpha = 0.75
    p_dict = {}
    for r in schedule.rig_dict.keys():
        dfr = schedule.gantt_df[schedule.gantt_df.rig.str.contains(r)]
        source = ColumnDataSource(dfr)
        rigs = list(dfr.rig.unique())
        rigs.reverse()
        p_dict[r] = figure(y_range=rigs, x_axis_type='datetime',
                           x_range=Range1d(schedule.gantt_start_date,
                                           schedule.gantt_end_date),
                           plot_width=1500, plot_height=plot_height,
                           toolbar_location='above', active_drag='pan',
                           title=r+' Schedule')
        p_dict[r].hbar(y='rig', left='start', right='end', color='color',
                       fill_alpha=fill_alpha, height=height,
                       line_color='gray', source=source)
        if r == 'DUC':
            hover = HoverTool(
                tooltips=[
                    ('Pad', '@pad'),
                    ('Start', '@start_text'),
                    ('Finish', '@end_text'),
                    ('Avg Compl Time', '@compl'),
                    ('Well Count', '@well_count')
                ]
            )
        else:
            hover = HoverTool(
                tooltips=[
                    ('Pad', '@pad'),
                    ('Start', '@start_text'),
                    ('Finish', '@end_text'),
                    ('Avg Drill Time', '@drill'),
                    ('Avg Compl Time', '@compl'),
                    ('Well Count', '@well_count')
                ]
            )
        labels = LabelSet(x='start', y='rig', text='pad', level='glyph',
                          angle=angle, angle_units=angle_units,
                          x_offset=x_offset, y_offset=-y_offset,
                          source=source, render_mode='canvas',
                          text_font_size=text_font_size,
                          text_font_style=text_font_style,
                          text_color='black')
        p_dict[r].add_tools(hover)
        p_dict[r].add_layout(labels)
        p_dict[r].ygrid.grid_line_color = None
        p_dict[r].xaxis.axis_label = "Date"
        p_dict[r].xaxis.formatter = DatetimeTickFormatter(
            months=["%b %Y"],
            days=["%b %e %Y"],
            years=["%b %Y"],
        )
        p_dict[r].outline_line_color = None
    p_columns = column(list(p_dict.values()))
    if schedule.show_gantt:
        show(p_columns)
        
    # else:
    #     save(p_columns)

def load_framework_scenario(framework):
    conn = connect(framework.branch.tree.connection_dict)
    query = str('select * from frameworks '
                'where scenario = \'' + framework.name + '\'')
    return pd.read_sql(query, conn).to_dict(orient='records')[0]

def end_date(framework):
    return (framework.effective_date + relativedelta(years=+framework.life) - relativedelta(days=+1))

def padding_df(df, padding):
    pad_df = pd.DataFrame({c: np.zeros(padding) for c in df.columns})
    pad_df.prod_date = pd.NaT
    return pad_df

def load_forecast_scenario(framework, properties=None):
    if properties is None:
        prop_list = ', '.join('\'{0}\''.format(p) for p in framework.branch.properties.propnum.unique())
    else:
        prop_list = ', '.join('\'{0}\''.format(p) for p in properties.propnum.unique())
    conn = connect(framework.branch.tree.connection_dict)
    query = str('select * from forecasts '
                'where scenario = \'' + framework.branch.scenario.forecast + '\' '
                'and idp in (' + prop_list + ')')
    return pd.read_sql(query, conn)

def load_forecast(framework, idp, prod_forecast_scenario, t_start=None, t_end=None, eff_date=None, end_date=None):
    conn = connect(framework.branch.tree.connection_dict)
    if t_start is not None and t_end is not None:
        query = str('select * from prod_forecasts '
                    'where idp = \'' + idp + '\' ' +
                    'and time_on >= ' + str(t_start) + ' ' +
                    'and time_on <= ' + str(t_end) + ' ' +
                    'and scenario = \'' + prod_forecast_scenario + '\' '
                    'order by time_on')
    if eff_date is not None and end_date is not None:
        query = str('select * from prod_forecasts '
                    'where idp = \'' + idp + '\' ' +
                    'and prod_date >= \'' + eff_date + '\' ' +
                    'and prod_date <= \'' + end_date + '\' ' +
                    'and scenario = \'' + prod_forecast_scenario + '\' '
                    'order by prod_date')
    return pd.read_sql(query, conn)

def load_type_curve(forecaster, type_curve):
    conn = connect(forecaster.branch.tree.connection_dict)
    query = str('select * from type_curves '
                'where type_curve = \'' + type_curve + '\'')
    return pd.read_sql(query, conn)

def load_economics_scenario(framework, properties=None):
    if properties is None:
        prop_list = ', '.join('\'{0}\''.format(p) for p in framework.branch.properties.propnum.unique())
    else:
        prop_list = ', '.join('\'{0}\''.format(p) for p in properties.propnum.unique())
    conn = connect(framework.branch.tree.connection_dict)
    
    query = str('select * from economics ' +
                'where economics.scenario = \'' +
                framework.branch.scenario.economics + '\' ' +
                'and economics.idp in (' + prop_list + ')')
    return pd.read_sql(query, conn)  

def load_price_deck(framework):
    conn = connect(framework.branch.tree.connection_dict)
    query = str('select * from pricing ' +
                'where price_deck = \'' +
                framework.branch.scenario.price_deck + '\'')
    return pd.read_sql(query, conn)

def econ_parser(param_name, param, effective_date, prod_start_date, end_date):
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le}
    tmp_econ_df = pd.DataFrame(columns=['prod_date', 'eomonth', param_name, 'unit'])
    date_range = pd.date_range(effective_date, end_date, freq='D')
    prod_start_date = pd.Timestamp(prod_start_date.date())
    tmp_econ_df.prod_date = date_range
    eomonth = []
    for d in date_range:
        day = calendar.monthrange(d.year, d.month)[1]
        eomonth.append(datetime.datetime(d.year, d.month, day))
    tmp_econ_df.eomonth = pd.to_datetime(pd.Series(eomonth))
    try:
        param = float(param)
        tmp_econ_df[param_name] = param
        tmp_econ_df['unit'] = 'per'
        tmp_econ_df.loc[tmp_econ_df.prod_date < prod_start_date, param_name] = 0.0
        return tmp_econ_df
    except:
        try:
            param = float(param.strip('%'))/100.
            tmp_econ_df[param_name] = param
            tmp_econ_df['unit'] = 'pct'
            tmp_econ_df.loc[tmp_econ_df.prod_date < prod_start_date, param_name] = 0.0
            return tmp_econ_df
        except:
            params = param.split(' ')
            _iter = enumerate(params)
            prior_date = None
            for i, p in _iter:
                if i == 0:

                    if '%' in p:
                        try:
                            start_val = float(p.strip('%'))/100.
                            start_unit = 'pct'
                            continue
                        except:
                            print('first value must be float, provided', p.strip('%'))
                            sys.stdout.flush()
                        return
                    else:
                        try:
                            start_val = float(p)
                            start_unit = 'per'
                            continue
                        except:
                            print('first value must be float, provided', p)
                            sys.stdout.flush()
                        return

                else:
                    if p == 'if':
                        try:
                            op_val = params[i+1]
                            date_val= params[i+2]
                            else_op = params[i+3]
                            other_val = params[i+4]
                        except:
                            print('missing values after if')
                            sys.stdout.flush()
                            return
                        if op_val not in ops.keys():
                            print('invalid operator')
                            sys.stdout.flush()
                            return
                        try:
                            date_val = pd.Timestamp(date_val)
                        except:
                            print('invalid date value', date_val)
                            sys.stdout.flush()
                            return
                        if else_op != 'else':
                            print('invalid syntax', else_op)
                            sys.stdout.flush()
                            return
                        if '%' in other_val:
                            try:
                                other_val = float(other_val.strip('%'))/100.
                                other_unit = 'pct'
                            except:
                                print('invalid alternate value', other_val.strip('%'))
                                sys.stdout.flush()
                                return
                        else:
                            try:
                                other_val = float(other_val)
                                other_unit = 'per'
                            except:
                                print('invalid alternate value', other_val)
                                sys.stdout.flush()
                                return
                        if not ops[op_val](prod_start_date, date_val):
                            start_val = other_val
                            start_unit = other_unit
                        tmp_econ_df[param_name] = start_val
                        tmp_econ_df.unit = start_unit
                        _ = next(_iter)
                        _ = next(_iter)
                        _ = next(_iter)
                        _ = next(_iter)

                    if p == 'until':
                        try:
                            date_val = params[i+1]
                            then_op = params[i+2]
                            next_val = params[i+3]
                        except:
                            print('missing values after until')
                            sys.stdout.flush()
                            return
                        try:
                            date_val = pd.Timestamp(date_val)
                            sys.stdout.flush()
                        except:
                            print('invalid date value', date_val)
                            sys.stdout.flush()
                            return
                        if then_op != 'then':
                            print('invalid syntax', then_op)
                            sys.stdout.flush()
                            return
                        if '%'  in next_val:
                            try:
                                next_val = float(next_val.strip('%'))/100.
                                next_unit = 'pct'
                            except:
                                print('invalid next value', next_val.strip('%'))
                                sys.stdout.flush()
                        else:
                            try:
                                next_val = float(next_val)
                                next_unit = 'per'
                            except:
                                print('invalid next value', next_val)
                                sys.stdout.flush()
                                return
                        mask = operator.le(date_range, date_val)
                        antimask = ~mask
                        if prior_date is not None:
                            mask = (operator.gt(date_range, prior_date) & operator.le(date_range, date_val))
                            antimask = operator.gt(date_range, date_val)
                        tmp_econ_df.loc[mask, param_name] = start_val
                        tmp_econ_df.loc[mask, 'unit'] = start_unit
                        tmp_econ_df.loc[antimask, param_name] = next_val
                        tmp_econ_df.loc[antimask, 'unit'] = next_unit
                        start_val = next_val
                        start_unit = next_unit
                        prior_date = date_val
                        _ = next(_iter)
                        _ = next(_iter)
                        _ = next(_iter)

                    if p == 'for':
                        try:
                            time_val = params[i+1]
                            unit_val = params[i+2]
                            then_op = params[i+3]
                            next_val = params[i+4]
                        except:
                            print('missing values after for')
                            sys.stdout.flush()
                        try:
                            time_val = int(time_val)
                        except:
                            print('invalid time value', time_val)
                            sys.stdout.flush()
                            return
                        if unit_val not in ('d', 'day', 'days',
                                            'mo', 'mos', 'month', 'months',
                                            'y', 'yr', 'yrs', 'year', 'years'):
                            print('unknown date unit', unit_val)
                            sys.stdout.flush()
                            return
                        if '%'  in next_val:
                            try:
                                next_val = float(next_val.strip('%'))/100.
                                next_unit = 'pct'
                            except:
                                print('invalid next value', next_val.strip('%'))
                                sys.stdout.flush()
                        else:
                            try:
                                next_val = float(next_val)
                                next_unit = 'per'
                            except:
                                print('invalid next value', next_val)
                                sys.stdout.flush()
                                return
                        if unit_val in ('d', 'day', 'days'):
                            delta = relativedelta(days=time_val)
                        if unit_val in ('m', 'mo', 'mos', 'month', 'months'):
                            delta = relativedelta(months=time_val)
                        if unit_val in ('y', 'yr', 'yrs', 'year', 'years'):
                            delta = relativedelta(years=time_val)
                        end_date = prod_start_date + delta
                        mask = date_range < end_date
                        tmp_econ_df.loc[mask, param_name] = start_val
                        tmp_econ_df.loc[mask, 'unit'] = start_unit
                        tmp_econ_df.loc[~mask, param_name] = next_val
                        tmp_econ_df.loc[~mask, 'unit'] = next_unit
                        start_val = next_val
                        start_unit = next_unit
                        _ = next(_iter)
                        _ = next(_iter)
                        _ = next(_iter)
                        _ = next(_iter)

    tmp_econ_df.loc[tmp_econ_df.prod_date < prod_start_date, param_name] = 0.0
    return tmp_econ_df

def misc_capex_parser(param, effective_date, end_date):
    tmp_df = pd.DataFrame(columns=['prod_date', 'eomonth', 'inv_g_misc'])
    date_range = pd.date_range(effective_date, end_date, freq='D')
    tmp_df.prod_date = date_range
    eomonth = []
    for d in date_range:
        day = calendar.monthrange(d.year, d.month)[1]
        eomonth.append(datetime.datetime(d.year, d.month, day))
    tmp_df.eomonth = pd.to_datetime(pd.Series(eomonth))
    tmp_df.inv_g_misc = 0.0
    params = param.split(',')
    for p in params:
        if p[0] == ' ':
            p = p[1:]
        p_split = p.split(' ')
        if len(p_split) == 1:
            try:
                p_split = float(p_split[0])
                if p_split > 0.001:
                    print('misc capex with no date provided', p_split)
                    sys.stdout.flush()
                    return tmp_df
                else:
                    return tmp_df
            except:
                print('bad misc capex', p_split)
                sys.stdout.flush()
                return tmp_df
        if len(p_split) > 1:
            try:
                p_cap = float(p_split[0])
            except:
                print('capex not float')
                sys.stdout.flush()
                return tmp_df
            if p_split[1] != 'on':
                print('invalid syntax, missing \'on\'')
                sys.stdout.flush()
                return tmp_df
            try:
                p_date = pd.Timestamp(p_split[2])
            except:
                print('invalid date')
                sys.stdout.flush()
                return tmp_df
            tmp_df.loc[tmp_df.prod_date == p_date, 'inv_g_misc'] = p_cap
    return tmp_df

def aban_capex_parser(param, end_of_life, effective_date, end_date):
    tmp_df = pd.DataFrame(columns=['prod_date', 'eomonth', 'inv_g_aban'])
    date_range = pd.date_range(effective_date, end_date, freq='D')
    tmp_df.prod_date = date_range
    eomonth = []
    for d in date_range:
        day = calendar.monthrange(d.year, d.month)[1]
        eomonth.append(datetime.datetime(d.year, d.month, day))
    tmp_df.eomonth = pd.to_datetime(pd.Series(eomonth))
    tmp_df.inv_g_aban = 0.0
    try:
        p_cap = float(param)
        tmp_df.loc[tmp_df.prod_date == end_of_life, 'inv_g_aban'] = p_cap
    except:
        sys.stdout.flush()
        params = param.split(' ')
        try:
            p_cap = float(params[0])
        except:
            print('capex not float')
            sys.stdout.flush()
        if params[1] != 'after':
            print('invalid syntax, missing \'after\'')
            sys.stdout.flush()
            return
        try:
            time_delta = int(params[2])
        except:
            print('bad time delta', params[2])
            sys.stdout.flush()
            return
        unit_val = params[3]
        if unit_val in ('d', 'day', 'days'):
            delta = relativedelta(days=time_delta)
        elif unit_val in ('m', 'mo', 'mos', 'month', 'months'):
            delta = relativedelta(months=time_delta)
        elif unit_val in ('y', 'yr', 'yrs', 'year', 'years'):
            delta = relativedelta(years=time_delta)
        else:
            print('invalid timestep', unit_val)
            sys.stdout.flush()
            return
        aban_date = pd.Timestamp(end_of_life) + delta
        tmp_df.loc[tmp_df.prod_date == aban_date, 'inv_g_aban'] = p_cap
    return tmp_df

def min_life_parser(param, first_neg_date, effective_date, end_date):
    tmp_df = pd.DataFrame(columns=['prod_date', 'eomonth', 'min_life'])
    date_range = pd.date_range(effective_date, end_date, freq='D')
    tmp_df.prod_date = date_range
    eomonth = []
    for d in date_range:
        day = calendar.monthrange(d.year, d.month)[1]
        eomonth.append(datetime.datetime(d.year, d.month, day))
    tmp_df.eomonth = pd.to_datetime(pd.Series(eomonth))
    tmp_df.min_life = 1
    if param in ('loss ok', 'lossok', 'loss_ok'):
        return tmp_df
    try:
        params = param.split(' ')
    except:
        print('invalid syntax, can\'t split', param)
        sys.stdout.flush()
        return
    try:
        time_delta = int(params[0])
    except:
        print('time value not int', params[0])
        sys.stdout.flush()
        return
    unit_val = params[1]
    if unit_val in ('d', 'day', 'days'):
        delta = relativedelta(days=time_delta)
    elif unit_val in ('m', 'mo', 'mos', 'month', 'months'):
        delta = relativedelta(months=time_delta)
    elif unit_val in ('y', 'yr', 'yrs', 'year', 'years'):
        delta = relativedelta(years=time_delta)
    else:
        print('invalid timestep', unit_val)
        sys.stdout.flush()
        return
    if len(params) > 2:
        if params[2] != 'after':
            print('invalid syntax, missing \'after\'')
            return                
        if len(params) == 5:
            p = params[3] + ' ' + params[4]
        else:
            p = params[3]
        if p in ('effective date', 'eff', 'eff date',
                    'effective_date', 'eff', 'eff_date'):
            min_date = effective_date + delta
            if first_neg_date < min_date:
                end_of_life = min_date
            else:
                end_of_life = first_neg_date
        if p == 'life':
            end_of_life = first_neg_date + delta
    else:
        min_date = effective_date + delta
        if first_neg_date < min_date:
            end_of_life = min_date
        else:
            end_of_life = first_neg_date
    tmp_df.loc[tmp_df.prod_date > end_of_life, 'min_life'] = 0
    return tmp_df

def load_output_from_sql(branch, scenario_name):
    conn = connect(branch.tree.connection_dict)
    start_date = branch.framework.effective_date
    end_date = branch.framework.end_date
    query = str('select * from output ' +
                'where scenario = \'' + scenario_name + '\' '
                'and prod_date >= \'' + start_date.strftime('%x') + '\' '
                'and prod_date <= \'' + end_date.strftime('%x') + '\'')
    return pd.read_sql(query, conn)    

def update_forecast(branch, w, production, first_on, last_on):
    conn = connect(branch.tree.connection_dict)
    eng = engine(branch.tree.connection_dict)
    cursor = conn.cursor()
    query = str('delete from prod_forecasts ' +
                'where scenario = \'' + w.prod_forecast_scenario + '\' '
                'and time_on >= ' + str(first_on) + ' '
                'and time_on <= ' + str(last_on) + ' '
                'and idp = \'' + w.idp + '\'')
    cursor.execute(query)
    conn.commit()
    cursor.close()
    print('deleted old production')
    production.to_sql(name='prod_forecasts', con=eng,
                      if_exists='append', method='multi',
                      index=False, chunksize=500)
    print('saved new production')

def update_cumulative(branch, w):
    conn = connect(branch.tree.connection_dict)
    eng = engine(branch.tree.connection_dict)
    cursor = conn.cursor()
    query = str('update prod_forecasts '
                'set cum_gas = b.cum_gas '
                'from prod_forecasts a inner join (select time_on, sum(gas) over (order by time_on) as cum_gas '
                'from prod_forecasts where idp = \'' + w.idp + '\' and scenario = \'' + w.prod_forecast_scenario + '\') b '
                'on a.time_on = b.time_on '
                'where idp = \'' + w.idp + '\' and scenario = \'' + w.prod_forecast_scenario + '\'')
    cursor.execute(query)
    print('updated gas cumulative production')
    conn.commit()
    query = str('update prod_forecasts '
                'set cum_oil = b.cum_oil '
                'from prod_forecasts a inner join (select time_on, sum(oil) over (order by time_on) as cum_oil '
                'from prod_forecasts where idp = \'' + w.idp + '\' and scenario = \'' + w.prod_forecast_scenario + '\') b '
                'on a.time_on = b.time_on '
                'where idp = \'' + w.idp + '\' and scenario = \'' + w.prod_forecast_scenario + '\'')
    cursor.execute(query)
    print('updated oil cumulative production')
    conn.commit()
    query = str('update prod_forecasts '
                'set cum_water = b.cum_water '
                'from prod_forecasts a inner join (select time_on, sum(water) over (order by time_on) as cum_water '
                'from prod_forecasts where idp = \'' + w.idp + '\' and scenario = \'' + w.prod_forecast_scenario + '\') b '
                'on a.time_on = b.time_on '
                'where idp = \'' + w.idp + '\' and scenario = \'' + w.prod_forecast_scenario + '\'')
    cursor.execute(query)
    conn.commit()
    print('updated water cumulative production')
    cursor.close()

def delete_output(framework):
    start = time.time()
    conn = connect(framework.branch.tree.connection_dict)
    eng = engine(framework.branch.tree.connection_dict)
    cursor = conn.cursor()
    if framework.rename is not None:
        scenario = framework.rename
    else:
        scenario = framework.branch.scenario.scenario
    if framework.delete_all:
        print('\ndeleting', scenario, 'from output table')
        sys.stdout.flush()
        query = str('delete from output '
                    'where scenario = \'' + scenario + '\'')
    else:
        prop_list = framework.branch.properties.propnum.unique()
        print('\ndeleting', len(prop_list), 'properties from output scenario', scenario)
        sys.stdout.flush()
        prop_list = ', '.join('\'{0}\''.format(p) for p in prop_list)
        query = str('delete from output '
                    'where scenario = \'' + scenario + '\' '
                    'and idp in (' + prop_list + ')')
    cursor.execute(query)
    conn.commit()
    cursor.close()
    stop = time.time()
    timer(start, stop)

def save_output(framework):
    start = time.time()
    conn = connect(framework.branch.tree.connection_dict)
    eng = engine(framework.branch.tree.connection_dict)
    if 'system' in framework.output.columns:
        placeholders = ', '.join('?' * (len(framework.output.columns) - 1))
    else:
        placeholders = ', '.join('?' * len(framework.output.columns))
    query = str('insert into output values (' + placeholders + ')')
    framework.output.loc[:, 'prod_date'] = framework.output.loc[:, 'prod_date'].dt.strftime('%')
    sys.stdout.flush()
    cursor = conn.cursor()
    if 'system' in framework.output.columns:
        cursor.executemany(query, list(framework.output.drop(columns=['system']).itertuples(index=False, name=None)))
    else:
       cursor.executemany(query, list(framework.output.itertuples(index=False, name=None)))
    conn.commit()
    cursor.close()
    stop = time.time()
    timer(start, stop)

def save_scenario(tree, scenario):
    conn = connect(tree.connection_dict)
    eng = engine(tree.connection_dict)
    query = str('insert into scenarios values (?, ?, ?, ?, ?, ?, ?, ?')
    cursor = conn.cursor()
    cursor.executemany(query, pd.DataFrame(scenario).itertuples(index=False, name=None))
    cursor.close()

def save_object(obj, pkl_name):
    with open(pkl_name+'.pkl', 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    
def load_object(name):
    if '.pkl' not in name:
        name = name + '.pkl'
    with open(name, 'rb') as input:
        return pickle.load(input)

def timer(start, stop):
    runtime = (stop - start)
    if runtime < 60:
        print('execution time:',
              round(runtime, 2), 'seconds')
        sys.stdout.flush()
    if runtime >= 60 and runtime < 3600:
        print('execution time:',
              round(runtime/60, 2), 'minutes')
        sys.stdout.flush()
    if runtime >= 3600:
        print('execution time:',
              round(runtime/3600, 2), 'hours')
        sys.stdout.flush()

def load_input_capacity_volumes(capacity):
    conn = connect(capacity.branch.tree.connection_dict)    
    query = str('select * from input_capacity_volumes ' +
                'where scenario = \'' + capacity.branch.scenario.capacity + '\'')
    return pd.read_sql(query, conn)

def load_max_system_volumes(capacity):
    conn = connect(capacity.branch.tree.connection_dict)    
    query = str('select * from max_system_volumes ' +
                'where scenario = \'' + capacity.branch.scenario.capacity + '\'')
    return pd.read_sql(query, conn)

def load_system_output(capacity):
    conn = connect(capacity.branch.tree.connection_dict)
    well_list = ', '.join('\'{0}\''.format(w) for w in capacity.branch.properties.propnum.unique())
    query = str('select propnum, short_pad, pad, system from properties ' +
                'where propnum in (' + well_list + ') '
                'and properties.scenario = \'' + capacity.branch.scenario.properties + '\'')
    return pd.read_sql(query, conn)

def initialize_capacity_dataframe(capacity):
    df = pd.DataFrame(columns=['system', 'prod_date', 'gross_gas', 'max_volume', 'delta'])
    num_systems = len(capacity.systems)
    date_range = capacity.framework.date_range
    num_days = len(date_range)
    df['system'] = capacity.systems * num_days
    df['prod_date'] = list(date_range) * num_systems
    df_list = []
    for system in capacity.systems:
        if system == 'STORY_GULCH':
            d = df[df.system == 'system'].copy()
            m = capacity.max_volumes[capacity.max_volumes.system == system].copy()
            s = capacity.system_volumes[capacity.system_volumes == system].copy()
            d.loc[:, 'max_volume'] = m.loc[:, 'max_volume'].values
            d.loc[:, 'gross_gas'] = s.groupby(by=['prod_date']).gross_gas.sum()
            d.loc[:, 'delta'] = d.loc[:, 'gross_gas'] - d.loc[:, 'max_volume']
        if system == 'K24':
            continue    

def save_to_excel(branch, file_path):
    try:
        if file_path == 'main':
            file_path = str(branch.tree.name + '\\' + branch.scenario.scenario +
                            '_capacity_model.xlsx')
    except PermissionError:
        print('file is open and cannot be overwritten, use the archive copy')
    if file_path == 'archive':
        file_path = str(branch.tree.name + '\\archive\\' + branch.scenario.scenario +
                        '_capacity_model_' + branch.tree.run_time + '.xlsx')
    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    for name, model in branch.capacity.models.items():
        model.to_excel(writer, sheet_name=name, index=False)
    writer.save()

def save_dists_to_excel(branch, file_path):
    name = branch.scenario.scenario
    try:
        if file_path == 'main':
            file_path = str(branch.tree.name + '\\' + branch.scenario.scenario +
                            '_probability_distributions.xlsx')
            writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
            branch.framework.econ_dists.to_excel(writer, sheet_name=name, index=False)
            writer.save()
    except PermissionError:
        print('file is open and cannot be overwritten, use the archive copy')
    if file_path == 'archive':
        file_path = str(branch.tree.name + '\\archive\\' + branch.scenario.scenario +
                        '_probability_distributions_' + branch.tree.run_time + '.xlsx')
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        branch.framework.econ_dists.to_excel(writer, sheet_name=name, index=False)
        writer.save()    

def save_dists_to_sql(branch):
    conn = connect(branch.tree.connection_dict)
    eng = engine(branch.tree.connection_dict)
    # cursor = conn.cursor()
    # query = str('delete from probability_distributions '
    #             'where scenario = \'' + branch.name + '\'')
    # cursor.execute(query)
    # conn.commit()
    # cursor.close()
    branch.framework.econ_dists['run_date'] = pd.Timestamp(branch.tree.run_time)
    branch.framework.econ_dists['scenario'] = branch.name
    branch.framework.econ_dists.to_sql(name='probability_distributions', con=eng,
                                       if_exists='append', method='multi',
                                       index=False, chunksize=500)

def save_aggregation_to_sql(branch, subset, label):
    conn = connect(branch.tree.connection_dict)
    eng = engine(branch.tree.connection_dict)
    cursor = conn.cursor()
    if label:
        query = str('delete from probability_aggregations '
                    'where scenario = \'' + branch.name + '_' + label + '\'')
    else:
        query = str('delete from probability_aggregations '
                    'where scenario = \'' + branch.name + '\'')        
    cursor.execute(query)
    conn.commit()
    cursor.close()
    if label:
        temp = pd.DataFrame(branch.framework.aggregations[subset])
    else:
        temp = pd.DataFrame(branch.framework.aggregations['All'])
    temp['run_date'] = pd.Timestamp(branch.tree.run_time)
    temp['scenario'] = branch.name
    temp.to_sql(name='probability_aggregations', con=eng,
                if_exists='append', method='multi',
                index=False, chunksize=500)

def save_aggregation_to_excel(branch, file_path, subset):
    name = branch.scenario.scenario
    try:
        if file_path == 'main':
            if subset:
                file_path = str(branch.tree.name + '\\' + branch.scenario.scenario +
                                '_probability_aggregations_' + subset + '.xlsx')
            else:
                file_path = str(branch.tree.name + '\\' + branch.scenario.scenario +
                                '_probability_aggregations.xlsx')
            writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
            if subset:
                branch.framework.aggregations[subset].to_excel(writer, sheet_name=name, index=False)
            else:
                branch.framework.aggregations['All'].to_excel(writer, sheet_name=name, index=False)
            writer.save()
    except PermissionError:
        print('file is open and cannot be overwritten, use the archive copy')
    if file_path == 'archive':
        if subset:
            file_path = str(branch.tree.name + '\\archive\\' + branch.scenario.scenario +
                            '_probability_aggregations_' + subset + '_' + branch.tree.run_time + '.xlsx')
        else:
            file_path = str(branch.tree.name + '\\archive\\' + branch.scenario.scenario +
                            '_probability_aggregations_' + branch.tree.run_time + '.xlsx')
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        if subset:
            branch.framework.aggregations[subset].to_excel(writer, sheet_name=name, index=False)
        else:
            branch.framework.aggregations['All'].to_excel(writer, sheet_name=name, index=False)
        writer.save()   

def tablemaker(table, prod_date, forecast, overflow, month1,month2,month3):
    month1_index = []
    month2_index = []
    month3_index = []
    for value in prod_date:
        if value[5:7] == str(month1):
            month1_index.append(prod_date.index(value))
        elif value[5:7] == str(month2):
            month2_index.append(prod_date.index(value))
        elif value[5:7] == str(month3):
            month3_index.append(prod_date.index(value))
    for i,j,k in zip(range(min(month1_index),max(month1_index)), range(min(month2_index),max(month2_index)), range(min(month3_index),max(month3_index))):
        try:
            table.add_row(prod_date[i], '{:,.0f}'.format(forecast[i]), '{:,.0f}'.format(overflow[i]),
                          prod_date[j], '{:,.0f}'.format(forecast[j]), '{:,.0f}'.format(overflow[j]),
                          prod_date[k], '{:,.0f}'.format(forecast[k]), '{:,.0f}'.format(overflow[k]))
        except:
            table.add_row(' ', ' ', ' ',
                          ' ', ' ',' ',
                          ' ', ' ', ' ')

        table.add_hline(cmidruleoption = '[3pt]')

def monthly_avg(df, month1, month2, month3):
    d = {month1: [[],[],[]],
             month2:  [[],[],[]],
             month3:  [[],[],[]]}
    d_avg = {month1: [], month2:  [],month3:  []}

    for index,row in df.iterrows():
        if row['prod_date'].month == month1:
            d[month1][0].append(row['forecast'])
            d[month1][1].append(row['max_volume'])
            d[month1][2].append(row['report_overflow'])
        if row['prod_date'].month == month2:
            d[month2][0].append(row['forecast'])
            d[month2][1].append(row['max_volume'])
            d[month2][2].append(row['report_overflow'])
        if row['prod_date'].month == month3:
            d[month3][0].append(row['forecast'])
            d[month3][1].append(row['max_volume'])
            d[month3][2].append(row['report_overflow'])

    for key,value in d.items():
        if key == month1:
            d_avg[month1].append(sum(value[0])/len(value[0]))
            d_avg[month1].append(sum(value[1])/len(value[1]))
            d_avg[month1].append(sum(value[2])/len(value[2]))
        if key == month2:
            d_avg[month2].append(sum(value[0])/len(value[0]))
            d_avg[month2].append(sum(value[1])/len(value[1]))
            d_avg[month2].append(sum(value[2])/len(value[2]))
        if key == month3:
            d_avg[month3].append(sum(value[0])/len(value[0]))
            d_avg[month3].append(sum(value[1])/len(value[1]))
            d_avg[month3].append(sum(value[2])/len(value[2]))
    return d_avg

def avg_tablemaker(table, df, month1, month2, month3):
    d_avg = monthly_avg(df, month1, month2, month3)
    row_cells = (MultiColumn(3, align='c|', data=datetime.date(1900, month1, 1).strftime('%B')),
                 MultiColumn(3, align='c|', data=datetime.date(1900, month2, 1).strftime('%B')),
                 MultiColumn(3, align='c', data=datetime.date(1900, month3, 1).strftime('%B')))
    table.add_row(row_cells)
    table.add_hline()
    table.add_row('Gas', 'Capacity', 'Delta',
                  'Gas', 'Capacity', 'Delta',
                  'Gas', 'Capacity', 'Delta')
    table.add_hline()
    table.add_row('{:,.0f}'.format(d_avg[month1][0]),
                  '{:,.0f}'.format(d_avg[month1][1]),
                  '{:,.0f}'.format(d_avg[month1][2]),
                  '{:,.0f}'.format(d_avg[month2][0]),
                  '{:,.0f}'.format(d_avg[month2][1]),
                  '{:,.0f}'.format(d_avg[month2][2]),
                  '{:,.0f}'.format(d_avg[month3][0]),
                  '{:,.0f}'.format(d_avg[month3][1]),
                  '{:,.0f}'.format(d_avg[month3][2]))

def well_properties_dataframe(branch):
    d = branch.schedule.well_dict
    for iw, w in enumerate(d):
        if iw == 0:
            df = pd.DataFrame(columns=d[w].__dict__.keys())
            well_lists = [[] for c in range(len(df.columns))]
            for ip, p in enumerate(d[w].__dict__.keys()):
                well_lists[ip].append(d[w].__dict__[p])
        else:
            for ip, p in enumerate(d[w].__dict__.keys()):
                well_lists[ip].append(d[w].__dict__[p])
    for ic, c in enumerate(well_lists):
        df.iloc[:, ic] = pd.Series(c)
    return df

def schedule_pad_properties_dataframe(branch):
    d = branch.schedule.pad_dict
    for iw, w in enumerate(d):
        if iw == 0:
            df = pd.DataFrame(columns=d[w].__dict__.keys())
            pad_lists = [[] for c in range(len(df.columns))]
            for ip, p in enumerate(d[w].__dict__.keys()):
                well_list = []
                if p == 'well_list':
                    for well in d[w].__dict__[p]:
                        well_list.append(well.well_name)
                    pad_lists[ip].append(' '.join(wn for wn in well_list))
                elif p == 'rig':
                    pad_lists[ip].append(d[w].__dict__[p].rig_name)
                else:
                    pad_lists[ip].append(d[w].__dict__[p])
        else:
            for ip, p in enumerate(d[w].__dict__.keys()):
                well_list = []
                if p == 'well_list':
                    for well in d[w].__dict__[p]:
                        well_list.append(well.well_name)
                    pad_lists[ip].append(' '.join(wn for wn in well_list))
                elif p == 'rig':
                    pad_lists[ip].append(d[w].__dict__[p].rig_name)
                else:
                    pad_lists[ip].append(d[w].__dict__[p])
    for ic, c in enumerate(pad_lists):
        df.iloc[:, ic] = pd.Series(c)
    return df

def framework_dataframe(branch):
    df = pd.DataFrame.from_dict(branch.framework.__dict__, orient='index')
    df.loc['schedule', :] = branch.framework.schedule.name
    df.loc['properties', :] = branch.framework.properties.scenario.unique()[0]
    df.loc['branch', :] = branch.name
    df.loc['tree', :] = branch.tree.name
    df.loc['date_range'] = (branch.framework.effective_date.strftime('%x') + ' ' +
                            branch.framework.end_date.strftime('%x'))
    df.drop(index=['scenario', 'economics', 'output'], inplace=True)
    if 'well_dict' in df.index:
        df.drop(index='well_dict', inplace=True)
    if 'price_deck' in df.index:
        df.drop(index='price_deck', inplace=True)
    return df

def xirr(cf):
    tol = 10
    disc_cf = cf.sum()
    max_iter = 1000

    if cf.sum() > 0:
        step = 0.01
        iter = 0
        guess = 0.0
        while abs(disc_cf) > tol:
            disc_cf = xnpv(cf, guess)
            if abs(disc_cf) > tol:
                if disc_cf > 0:
                    guess += step
                else:
                    guess -= step
                    step /= 2.0
                iter += 1
            if iter == max_iter:
                break
        return guess
    else:
        step = -0.01
        iter = 0
        guess = 0.0
        while abs(disc_cf) > tol:
            disc_cf = xnpv(cf, guess)
            if abs(disc_cf) > tol:
                if disc_cf > 0:
                    guess -= step
                    step /= 2.0
                else:
                    guess += step
                iter += 1
            if iter == max_iter:
                break
            if guess <= -1.0:
                return -1
        return guess

def npv(cf, d=0.1):
    return cf/(1+d)**(np.arange(cf.shape[0])/365.25)

def xnpv(cf, d=0.1):
    return sum(cf/(1+d)**(np.arange(cf.shape[0])/365.25))

def run_export(tree, file_path):
    b = tree.branches[list(tree.branches.keys())[0]]
    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    pd.DataFrame.from_dict(b.scenario, orient='index').to_excel(writer, sheet_name='scenarios',
                                                                header=False)
    if b.schedule is not None:
        b.schedule.schedule_df.to_excel(writer, sheet_name=b.scenario.scenario,
                                        index=False)
        b.schedule.schedule_inputs.to_excel(writer, sheet_name=b.scenario.schedule_inputs,
                                            index=False)
        b.schedule.gantt_df.to_excel(writer, sheet_name='gantt_chart', index=False)
        well_df = well_properties_dataframe(b)
        well_df.to_excel(writer, sheet_name='schedule_well_properties', index=False)
        pad_df = schedule_pad_properties_dataframe(b)
        pad_df.to_excel(writer, sheet_name='schedule_pad_properties', index=False)
    if b.framework is not None:
        framework_df = framework_dataframe(b)
        framework_df.to_excel(writer, sheet_name='framework', header=False)
        b.framework.properties.to_excel(writer, sheet_name='properties', index=False)
    if b.framework.economics is not None:
        b.framework.economics.to_excel(writer, sheet_name='economics', index=False)
        b.framework.price_deck.to_excel(writer, sheet_name='price_deck', index=False)
        well_df = well_properties_dataframe(b)
        well_df.to_excel(writer, sheet_name='framework_well_properties', index=False)
    if b.forecaster is not None:
        b.forecaster.well_dict
    if b.capacity:
        if b.capacity.models:
            b.capacity.input_volumes.to_excel(writer, sheet_name='capacity_input_volumes', index=False)
            b.capacity.max_volumes.to_excel(writer, sheet_name='capacity_max_volumes', index=False)
            b.capacity.systems.to_excel(writer, sheet_name='systems', index=False)
    writer.save()

def load_auto_params(forecaster):
    connection = connect(forecaster.branch.tree.connection_dict)
    query = str('select * from autoforecaster '
                'where scenario = \'' + forecaster.branch.scenario.autoforecaster + '\'')
    auto_df = pd.read_sql(query, connection)

    auto_params = {'gas': None,
                   'oil': None,
                   'water': None}

    for t in auto_df.prod_type.unique():
        bounds = {}
        initial_guesses = [0., 0., 0.]
        dmin = 0.25
        min_rate = 0.0

        for idx, row in auto_df[auto_df.prod_type == t].iterrows():
            if row.days_on not in bounds.keys():
                bounds[row.days_on] = ([0., 0., 0.], [0., 0., 0.])
            if row.type == 'initial_b':
                initial_guesses[0] = float(row.value)
            if row.type == 'initial_di':
                initial_guesses[1] = float(row.value)
            if row.type == 'initial_ip':
                initial_guesses[2] = float(row.value)
            if row.type == 'lower_b':
                bounds[row.days_on][0][0] = float(row.value)
            if row.type == 'lower_di':
                bounds[row.days_on][0][1] = float(row.value)
            if row.type == 'lower_ip':
                bounds[row.days_on][0][2] = float(row.value)
            if row.type == 'upper_b':
                bounds[row.days_on][1][0] = float(row.value)
            if row.type == 'upper_di':
                bounds[row.days_on][1][1] = float(row.value)
            if row.type == 'upper_ip':
                bounds[row.days_on][1][2] = float(row.value)
            if row.type == 'dmin':
                dmin = float(row.value)
            if row.type == 'min_rate':
                min_rate = float(row.value)

        auto_params[t] = {'bounds': bounds,
                         'initial_guesses': initial_guesses,
                         'dmin': dmin,
                         'min_rate': min_rate}

    return auto_params

def load_prod_info(framework):
    connection = connect(framework.branch.tree.connection_dict)
    query = str('select * from prod_forecasts_info '
                'where scenario = \'' + framework.branch.scenario.forecast  + '\'')
    return pd.read_sql(query, connection)

def load_production(branch, properties):
    connection = connect(branch.tree.connection_dict)
    if isinstance(properties, list):
        prop_list = ', '.join('\'{0}\''.format(p) for p in properties)
    elif isinstance(properties, str):
        prop_list = str('\'' + properties + '\'')
    query = str('select * from daily_production '
                'where idp in (' + prop_list  + ') '
                'and prod_date <= getdate()')
    return pd.read_sql(query, connection)

def load_prod_forecast(branch, properties, time_on):
    connection = connect(branch.tree.connection_dict)
    if isinstance(properties, list):
        prop_list = ', '.join('\'{0}\''.format(p) for p in properties)
    elif isinstance(properties, str):
        prop_list = str('\'' + properties + '\'')
    query = str('select * from prod_forecasts '
                'where idp in (' + prop_list  + ') '
                'and time_on >= ' + time_on + '\'')
    return pd.read_sql(query, connection)

def arps_fit(params, dmin, min_rate):
    if abs(params[0] - 1) < 0.0001:
        params[0] = 0.999
    ai = ((1/params[0])*(np.power((1-params[1]), -params[0])-1))/365
    df = 1-np.power((1+params[0]*-np.log(1-dmin)), -1/params[0])
    af = (1/params[0])*(np.power((1-df), -params[0])-1)/365
    t = int((ai-af)/(params[0]*ai*af))
    m = np.arange(1, t+1)
    m_exp = np.arange(1, 45625-t)
    q = params[2]/np.power((1+params[0]*ai*m), 1/params[0])
    qf = np.insert(q, 0, params[2])
    n = (np.power(params[2], params[0])*(np.power(params[2],
         (1-params[0]))-np.power(qf, (1-params[0])))/((1-params[0])*ai))
    forecast_arps = np.diff(n, axis=0)
    q0_exp = params[2]/np.power((1+params[0]*ai*t), 1/params[0])
    qf_exp = q0_exp*np.exp(-af*m_exp)
    n_exp = (q0_exp-qf_exp)/af
    forecast_exp = np.diff(n_exp, axis=0)
    forecast = np.concatenate([forecast_arps, forecast_exp])
    if any(forecast < min_rate):
        end_of_life = np.argmax(forecast < min_rate)
    else:
        end_of_life = len(forecast)
    forecast = forecast[:end_of_life]
    if len(forecast) < 18250:
        np.concatenate([forecast, np.zeros(18250-len(forecast))])

    return forecast

def residuals(params, y, dmin, min_rate, method='beta'):
    fcst = arps_fit(params, dmin, min_rate)[:len(y)]
    if len(fcst) < len(y):
        fcst = np.concatenate([fcst, np.zeros(len(y) - len(fcst))])
    if method == 'beta':
        beta_x = np.linspace(0.01, 0.99, len(y))
        cost = np.multiply(y - fcst, beta.pdf(beta_x, .98, 0.8))
        cost = np.divide(cost[y > 0], y[y > 0])
    if method == 'diff':
        cost = y - fcst
    if method == 'frac':
        cost = y / fcst
    return cost

def delete_prod_info(forecaster, overwrite, forecast_type):
    start = time.time()
    prop_list = list(forecaster.branch.properties.propnum.unique())
    num_prop = len(prop_list)
    if forecast_type:
        filtered_props = []
        for p in prop_list:
            if 'autotype' in forecaster.branch.model[p].forecasts.forecast_type:
                p_type = 'autotype'
            else:
                p_type = forecaster.branch.model[p].forecasts.forecast_type
            if isinstance(forecast_type, list):
                if p_type in forecast_type:
                    filtered_props.append(p)
            elif p_type == forecast_type:
                filtered_props.append(p)
        prop_list = filtered_props
        num_prop = len(prop_list)
    if not overwrite:
        connection = connect(forecaster.branch.tree.connection_dict)
        query = str('select distinct idp from prod_forecasts '
                    'where scenario = \'' + forecaster.branch.scenario.forecast + '\'')
        idp_list = pd.read_sql(query, connection)['idp'].values
        prop_list = [idp for idp in prop_list if idp not in idp_list]
        num_prop = len(prop_list)
    if prop_list:
        print('deleting', num_prop, 'forecast info onelines')
        sys.stdout.flush()
        prop_list = ', '.join('\'{0}\''.format(p) for p in prop_list)
        conn = connect(forecaster.branch.tree.connection_dict)
        eng = engine(forecaster.branch.tree.connection_dict)
        cursor = conn.cursor()
        query = str('delete from prod_forecasts_info '
                    'where scenario = \'' + forecaster.branch.scenario.forecast + '\' '
                    'and idp in (' + prop_list + ')')
        print(num_prop, 'total forecast info onelines deleted')
        cursor.execute(query)
        conn.commit()
        cursor.close()
    else:
        print('no properties deleted')
    stop = time.time()
    timer(start, stop)
    return

def save_prod_info(forecaster, overwrite):
    start = time.time()
    if not overwrite:
        connection = connect(forecaster.branch.tree.connection_dict)
        prop_list = forecaster.branch.properties.propnum.unique()
        query = str('select distinct idp from prod_forecasts_info '
                    'where scenario = \'' + forecaster.branch.scenario.forecast + '\'')
        idp_list = pd.read_sql(query, connection)['idp'].values
        save_properties = [idp for idp in prop_list if idp not in idp_list]
        prop_list = ', '.join('\'{0}\''.format(p) for p in save_properties)
    else:
        prop_list = ', '.join('\'{0}\''.format(p) for p in forecaster.branch.properties.propnum.unique())
    if prop_list and forecaster.prod_info['scenario']:
        conn = connect(forecaster.branch.tree.connection_dict)
        eng = engine(forecaster.branch.tree.connection_dict)
        placeholders = ', '.join('?' * len(forecaster.prod_info.keys()))
        query = str('insert into prod_forecasts_info values (' + placeholders + ')')
        cursor = conn.cursor()
        temp = pd.DataFrame(forecaster.prod_info)
        temp.dropna(inplace=True)
        temp.fillna(0, inplace=True)
        cursor.executemany(query, temp.itertuples(index=False, name=None))
        conn.commit()
        cursor.close()
        del temp
    else:
        print('no properties saved')
    stop = time.time()
    timer(start, stop)
    return

def save_manual_prod_info(branch, tmp_prod_info, prod_type, idp):
    start = time.time()
    conn = connect(branch.tree.connection_dict)
    eng = engine(branch.tree.connection_dict)
    query = str('delete from prod_forecasts_info '
                'where scenario = \'' + branch.scenario.forecast + '\' '
                'and prod_type = \'' + prod_type + '\' '
                'and idp = \'' + idp + '\'')
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    vals = '\'' + prod_type + '\', '
    cols = 'prod_type, '
    for k, v in tmp_prod_info.items():
        if 'date' in k:
            cols = cols + k + ', '
            vals += '\'' + pd.Timestamp(v[0]).strftime('%m/%d/%Y') + '\', '
        else:
            cols = cols + k + ', '
            if isinstance(v[0], str):
                vals += '\'' + str(v[0]) + '\', '
            else:
                vals += str(v[0]) + ', '
    vals = vals[:-2]
    cols = cols[:-2]
    query = str('insert into prod_forecasts_info (' + cols + ') values (' + vals + ')')
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    stop = time.time()
    timer(start, stop)
    return

def save_log(forecaster):
    start = time.time()
    if forecaster.log:
        conn = connect(forecaster.branch.tree.connection_dict)
        eng = engine(forecaster.branch.tree.connection_dict)
        placeholders = ', '.join('?' * len(forecaster.log.keys()))
        query = str('insert into prod_forecasts_log values (' + placeholders + ')')
        cursor = conn.cursor()
        cursor.executemany(query, pd.DataFrame(forecaster.log).itertuples(index=False, name=None))
        conn.commit()
        cursor.close()
    else:
        print('no log saved')
    stop = time.time()
    timer(start, stop)
    return

def properties_with_forecasts(forecaster):
    connection = connect(forecaster.branch.tree.connection_dict)
    prop_list = ', '.join('\'{0}\''.format(p) for p in forecaster.branch.properties.propnum.unique())
    query = str('select distinct idp from prod_forecasts '
                'where scenario = \'' + forecaster.branch.scenario.forecast + '\'')
    return pd.read_sql(query, connection)['idp'].values

def delete_prod_forecasts(forecaster, overwrite, forecast_type):
    start = time.time()
    prop_list = list(forecaster.branch.properties.propnum.unique())
    if forecast_type:
        filtered_props = []
        for p in prop_list:
            if 'autotype' in forecaster.branch.model[p].forecasts.forecast_type:
                p_type = 'autotype'
            else:
                p_type = forecaster.branch.model[p].forecasts.forecast_type
            if isinstance(forecast_type, list):
                if p_type in forecast_type:
                    filtered_props.append(p)
            elif p_type == forecast_type:
                filtered_props.append(p)
        prop_list = filtered_props
    if not overwrite:
        connection = connect(forecaster.branch.tree.connection_dict)
        query = str('select distinct idp from prod_forecasts '
                    'where scenario = \'' + forecaster.branch.scenario.forecast + '\'')
        idp_list = pd.read_sql(query, connection)['idp'].values
        prop_list = [idp for idp in prop_list if idp not in idp_list]
    if prop_list:
        print('deleting', len(prop_list), 'forecasts')
        sys.stdout.flush()
        prop_list = ', '.join('\'{0}\''.format(p) for p in prop_list)
        conn = connect(forecaster.branch.tree.connection_dict)
        eng = engine(forecaster.branch.tree.connection_dict)
        cursor = conn.cursor()
        query = str('delete from prod_forecasts '
                    'where scenario = \'' + forecaster.branch.scenario.forecast + '\' '
                    'and idp in (' + prop_list + ')')
        cursor.execute(query)
        conn.commit()
        cursor.close()
    else:
        print('no properties deleted')
    stop = time.time()
    timer(start, stop)
    return

def save_prod_forecasts(forecaster, fits):
    start = time.time()
    conn = connect(forecaster.branch.tree.connection_dict)
    eng = engine(forecaster.branch.tree.connection_dict)
    placeholders = ', '.join('?' * len(fits.keys()))
    query = str('insert into prod_forecasts values (' + placeholders + ')')
    cursor = conn.cursor()
    cursor.executemany(query, fits.itertuples(index=False, name=None))
    conn.commit()
    cursor.close()
    stop = time.time()
    timer(start, stop)
    return

def save_manual_prod_forecast(branch, fcst_dict, idp):
    start = time.time()
    conn = connect(branch.tree.connection_dict)
    eng = engine(branch.tree.connection_dict)
    query = str('delete from prod_forecasts '
                'where scenario = \'' + branch.scenario.forecast + '\' '
                'and idp = \'' + idp + '\'')
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    placeholders = ', '.join('?' * len(fcst_dict.keys()))
    query = str('insert into prod_forecasts values (' + placeholders + ')')
    temp = pd.DataFrame(fcst_dict)
    cursor = conn.cursor()
    cursor.executemany(query, temp.itertuples(index=False, name=None))
    conn.commit()
    cursor.close()
    del temp
    stop = time.time()
    timer(start, stop)
    return

def update_yields(forecaster, yields):
    start = time.time()
    conn = connect(forecaster.branch.tree.connection_dict)
    eng = engine(forecaster.branch.tree.connection_dict)
    yields.to_sql(name='temp_yields', con=eng,
                  if_exists='replace', method='multi',
                  index=False, chunksize=500)

    cursor = conn.cursor()
    query = str('update forecasts '
                'set gas_g_mpb = temp_yields.gas_g_mpb, '
                'oil_g_bpmm = temp_yields.oil_g_bpmm, '
                'water_g_bpmm = temp_yields.water_g_bpmm '
                'from forecasts inner join temp_yields '
                'on forecasts.idp = temp_yields.idp '
                'and forecasts.scenario = temp_yields.scenario')
    cursor.execute(query)
    conn.commit()

    query = str('drop table temp_yields')
    cursor.execute(query)
    conn.commit()

    cursor.close()
    stop = time.time()
    timer(start, stop)
    return

def join_dict(list_of_dicts):
    if isinstance(list_of_dicts, list):
        first = True
        d0 = None
        for i, d in enumerate(list_of_dicts):
                if d:
                    if first:
                        d0 = d.copy()
                        first = False
                    else:
                        for k, v in d.items():
                            d0[k] = np.concatenate([d0[k], v])
        return d0
    else:
        print('not a list')
        return None

def prob_dict(u_val):
    d = {}
    u_val = u_val.split(',')
    for i in u_val:
        if i[0] == ' ':
            i = i[1:]
        i = i.split(':')
        try:
            val = float(i[1])
        except:
            val = i[1]
        d[i[0]] = val
    return d

def load_probability_scenario(branch):
    connection = connect(branch.tree.connection_dict)
    query = str('select * from probabilities '
                'where scenario = \'' + branch.scenario.probability  + '\'')
    return pd.read_sql(query, connection)

def apply_uncertainty(d):
    if d['distribution'] == 'normal':
        a = (d['min'] - d['mean']) / d['stdev']
        b = (d['max'] - d['mean']) / d['stdev']
        return truncnorm.rvs(a, b, loc=d['mean'], scale=d['stdev'])
    if d['distribution'] == 'uniform':
        return np.random.uniform(d['min'], d['max'])
    if d['distribution'] == 'constant':
        return d['value']
    if d['distribution'] == 'coinflip':
        if np.random.random() <= d['probability']:
            return d['heads']
        else:
            return d['tails']

def apply_risk(d):
    p = d['probability']
    if np.random.random() <= p:
        if 'distribution' in d.keys():
            if d['distribution'] == 'normal':
                a = (d['min'] - d['mean']) / d['stdev']
                b = (d['max'] - d['mean']) / d['stdev']
                return truncnorm.rvs(a, b, loc=d['mean'], scale=d['stdev'])
            if d['distribution'] == 'uniform':
                return np.random.uniform(d['min'], d['max'])
            if d['distribution'] == 'constant':
                return d['value']
        elif 'cost' in d.keys():
            return d['cost']
        elif 'ip_mult' in d.keys():
            return d['ip_mult']
        elif 'tc_mult' in d.keys():
            return d['tc_mult']
    else:
        return None

def apply_curtailment(m, f):
    profile = f * m
    idx = np.nonzero(f)[0]
    if len(idx) == 0:
        return profile
    else:
        p = idx[0]
        t = int(5514.8*np.exp(-3.978*m))
        if p+45 > len(profile) or p+t > len(f):
            return profile
        else:
            v1 = profile[p+45]
            v2 = f[p+t]
            delta = (v2 - v1) / (t - 45)
            fill = np.arange(1, t-44) * delta + v1
            profile[p+45:p+t] = fill
            profile[p+t:] = f[p+t:]
            q1 = sum(f)
            q2 = sum(profile)
            delta_q = (q1 - q2) / 18160
            profile[p+90:] = profile[p+90:] + delta_q
            return profile

def apply_ip_adjust(m, f):
    profile = f * m
    idx = np.nonzero(f)[0]
    if len(idx) == 0:
        return profile
    else:
        p = idx[0]
        if m <= 1.0:
            x = np.linspace(0.1, 1, 19)
            y = [13904, 7966, 5285, 3832, 2919,
                2299, 1853, 1519, 1260, 1055,
                887, 752, 635, 533, 445,
                365, 293, 212, 0]
            t = int(np.interp(m, x, y))
            if p+90 > len(profile) or p+t > len(f):
                return profile
            else:
                v1 = profile[p+90]
                v2 = f[p+t]
                delta = (v2 - v1) / (t - 90)
                fill = np.arange(1, t-89) * delta + v1
                profile[p+90:p+t] = fill
                profile[p+t:] = f[p+t:]
                q1 = sum(f)
                q2 = sum(profile)
                delta_q = (q1 - q2) / 18160
                profile[p+90:] = profile[p+90:] + delta_q
                return profile
        else:
            if p+90 > len(profile) or p+t > len(f):
                return profile
            else:        
                v1 = profile[p+90]
                v2 = profile[p+120]
                delta = (v2 - v1) / 30
                fill = np.arange(1, 31) * delta + v1
                q1 = sum(f[p:p+120])
                q2 = sum(profile[p:p+120])
                delta_q = (q1 - q2) / 18160
                profile[p+90:p+120] = fill
                profile[p+120:] = profile[p+120:] + delta_q
                return profile

def build_agg_plot(branch, df, label):
    plt.figure(figsize=(5, 6))
    ax = sns.boxenplot(x=df.columns[0], y=df.columns[1], data=df, width=0.6)
    label_dict= {'gas_eur': 'Gas EUR',
                'ip90': 'First 90 Days Production',
                'drill_cost': 'Drill Capex',
                'compl_cost': 'Compl Capex',
                'irr': 'IRR',
                'npv': 'PV10',
                'payout': 'Payout',
                'year_1_fcf': 'First Year FCF',
                'year_1_roic': 'First Year ROIC',
                'year_2_fcf': 'Second Year FCF',
                'year_2_roic': 'Second Year ROIC'}

    if df.columns[1] in ('irr', 'year_1_roic', 'year_2_roic'):
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ymax = math.ceil(df[df.columns[1]].max()*10)/10
        if ymax < 0.5:
            y_range = np.linspace(0, ymax, (ymax/.02)+1)
        elif ymax < 1.5:
            y_range = np.linspace(0, ymax, (ymax/.05)+1)
        else:
            y_range = np.linspace(0, ymax, (ymax/.1)+1)
        plt.yticks(y_range)
        plt.ylabel(label_dict[df.columns[1]])
        plt.xlabel('')
        plt.tight_layout()
        if label:
            print('saving plot for ' + label_dict[df.columns[1]], label)
            file_path = str(branch.tree.name + '\\' + branch.scenario.scenario + '_' + df.columns[1] + '_' + label + '_plot')
            plt.savefig(file_path)
        else:
            print('saving plot for ' + label_dict[df.columns[1]])
            file_path = str(branch.tree.name + '\\' + branch.scenario.scenario + '_' + df.columns[1] + '_plot')
            plt.savefig(file_path)
        plt.close()
        return
    else:
        formatter = ticker.StrMethodFormatter('{x:,.0f}')
        ax.yaxis.set_major_formatter(formatter)
        plt.ylabel(label_dict[df.columns[1]])
        plt.xlabel('')
        plt.tight_layout()
        if label:
            print('saving plot for ' + label_dict[df.columns[1]], label)
            file_path = str(branch.tree.name + '\\' + branch.scenario.scenario + '_' + df.columns[1] + '_' + label + '_plot')
            plt.savefig(file_path)
        else:
            print('saving plot for ' + label_dict[df.columns[1]])
            file_path = str(branch.tree.name + '\\' + branch.scenario.scenario + '_' + df.columns[1] + '_plot')
            plt.savefig(file_path)
        plt.close()
        return

def event(l):
    return int(-math.log(1.0 - random.random()) / l)

def event_list(l, d, n):
    events = np.zeros(len(n), dtype=bool)
    e = event(l)
    d = int(d)
    for i in n:
        if i == e:
            if e + d > len(n):
                events[e:] = 1
            else:
                events[e:e+d] = 1
            e = event(l)
            if e < d:
                e = int(i) + d + 1
            else:
                e = int(i) + e
    return events

def run_query(branch, filters, updates):
    conn = connect(branch.tree.connection_dict)
    eng = engine(branch.tree.connection_dict)
    prop_list = ', '.join('\'{0}\''.format(p) for p in branch.properties.propnum.unique())
    start = time.time()
    for table in updates.keys():
        if table not in ('properties', 'economics', 'forecasts'):
            print('only the properties, economics, and forecasts tables can be updated')
        print('\nupdating table', table)
        update_query = 'update ' + table + ' set '
        select_query = 'select propnum, '
        for column, value in updates[table].items():
            if value[1] == 'string':
                update_query = update_query + column + ' = \'' + str(value[0]) + '\', '
            if value[1] == 'float':
                update_query = update_query + column + ' = ' + str(value[0]) + ', '
            if value[1] == 'literal':
                update_query = update_query + column + ' = ' + value[0] + ', '
            select_query = select_query + column + ', '
        update_query = update_query[:-2]
        update_query = update_query + ' from properties inner join economics on properties.propnum = economics.idp'
        update_query = update_query + ' inner join forecasts on properties.propnum = forecasts.idp'
        update_query = update_query + ' where properties.scenario = \'' + branch.scenario.properties + '\''
        update_query = update_query + ' and economics.scenario = \'' + branch.scenario.economics + '\''
        update_query = update_query + ' and forecasts.scenario = \'' + branch.scenario.forecast + '\''
        update_query = update_query + ' and properties.propnum in (' + prop_list + ')'
        select_query = select_query[:-2]
        select_query = select_query + ' from properties inner join economics on properties.propnum = economics.idp'
        select_query = select_query + ' inner join forecasts on properties.propnum = forecasts.idp'
        select_query = select_query + ' where properties.scenario = \'' + branch.scenario.properties + '\''
        select_query = select_query + ' and economics.scenario = \'' + branch.scenario.economics + '\''
        select_query = select_query + ' and forecasts.scenario = \'' + branch.scenario.forecast + '\''
        select_query = select_query + ' and properties.propnum in (' + prop_list + ')'
        if filters is not None:
            for tbl in filters.keys():
                for column, value in filters[tbl].items():
                    if value[2] == 'string':
                        update_query = update_query + ' and ' + tbl + '.' + column + ' ' + str(value[0]) + ' \'' + str(value[1]) + '\''
                        select_query = select_query + ' and ' + tbl + '.' + column + ' ' + str(value[0]) + ' \'' + str(value[1]) + '\''
                    if value[2] == 'float':
                        update_query = update_query + ' and ' + tbl + '.' + column + ' ' + str(value[0]) + ' ' + str(value[1])
                        select_query = select_query + ' and ' + tbl + '.' + column + ' ' + str(value[0]) + ' ' + str(value[1])
                    if value[2] == 'literal':
                        update_query = update_query + ' and ' + tbl + '.' + column + ' ' + str(value[0]) + ' ' + value[1]
                        select_query = select_query + ' and ' + tbl + '.' + column + ' ' + str(value[0]) + ' ' + value[1]
        old_data = pd.read_sql(select_query, conn)
        old_data = old_data.set_index('propnum').stack().reset_index()
        old_data.rename(columns={'level_1': 'variable', 0: 'value'}, inplace=True)
        old_data['change'] = 'original'
        old_data['table_name'] = table

        cursor = conn.cursor()
        cursor.execute(update_query)
        conn.commit()
        cursor.close()

        new_data = pd.read_sql(select_query, conn)
        new_data = new_data.set_index('propnum').stack().reset_index()
        new_data.rename(columns={'level_1': 'variable', 0: 'value'}, inplace=True)
        new_data['change'] = 'update'
        new_data['table_name'] = table
        
        change_log = pd.concat([old_data, new_data])
        if table == 'properties':
            scenario = branch.scenario.properties
        if table == 'economics':
            scenario = branch.scenario.economics
        if table == 'forecasts':
            scenario = branch.scenario.forecast
        change_log['scenario'] = scenario
        change_log['updated_by'] = os.getlogin()
        change_log['updated_on'] = pd.Timestamp(datetime.datetime.now())
        change_log['uuid'] = str(uuid.uuid1())

        insert_query = ('insert into change_log ([propnum], [variable], [value], [change], ' +
                        '[table_name], [scenario], [updated_by], [updated_on], [uuid])')
        insert_query = insert_query + ' values (?, ?, ?, ?, ?, ?, ?, ?, ?)'
        cursor = conn.cursor()
        cursor.executemany(insert_query, change_log.itertuples(index=False, name=None))
        conn.commit()
        cursor.close()

        stop = time.time()
        timer(start, stop)
    return

def run_restore_query(branch, uuid_list):
    conn = connect(branch.tree.connection_dict)
    eng = engine(branch.tree.connection_dict)
    if type(uuid_list) != list:
        uuid_list = [uuid_list]
    for uid in uuid_list:
        print('restoring inputs for', uid)
        query = ('select propnum, variable, ' +
                 'value, change, table_name, ' +
                 'scenario from change_log ' +
                 'where change = \'original\' and uuid = \'' + uid + '\'')
        new_data = pd.read_sql(query, conn)
        table_name = new_data.table_name.unique()[0]
        scenario = new_data.scenario.unique()[0]
        columns = new_data.variable.unique()
        df = new_data[['propnum', 'variable', 'value']]
        num_prop = len(df.propnum.unique())

        prop_list = ', '.join('\'{0}\''.format(p) for p in new_data.propnum.unique())
        if table_name == 'properties':
            change_query = 'select propnum, '
        else:
            change_query = 'select idp, '
        for c in columns:
            change_query = change_query + c + ', '
        change_query = change_query[:-2]
        change_query = change_query + ' from ' + table_name + ' where scenario = \'' + scenario + '\''
        if table_name == 'properties':
            change_query = change_query + ' and propnum in (' + prop_list + ')'
        else:
            change_query = change_query + ' and idp in (' + prop_list + ')'
        old_data = pd.read_sql(change_query, conn)
        old_data.rename(columns={'idp': 'propnum'}, inplace=True)
        old_data = old_data.set_index('propnum').stack().reset_index()
        old_data.rename(columns={'level_1': 'variable', 0: 'value'}, inplace=True)
        old_data['change'] = 'original'
        old_data['table_name'] = table_name
        if table_name == 'properties':
            scenario = branch.scenario.properties
        if table_name == 'economics':
            scenario = branch.scenario.economics
        if table_name == 'forecasts':
            scenario = branch.scenario.forecast
        old_data['scenario'] = scenario
        new_data['change'] = 'restore'
        change_log = pd.concat([old_data, new_data])
        change_log['updated_by'] = os.getlogin()
        change_log['updated_on'] = pd.Timestamp(datetime.datetime.now())
        change_log['uuid'] = str(uuid.uuid1())

        insert_query = ('insert into change_log ([propnum], [variable], [value], [change], ' +
                        '[table_name], [scenario], [updated_by], [updated_on], [uuid])')
        insert_query = insert_query + ' values (?, ?, ?, ?, ?, ?, ?, ?, ?)'
        cursor = conn.cursor()
        cursor.executemany(insert_query, change_log.itertuples(index=False, name=None))
        conn.commit()
        cursor.close()

        for i, p in enumerate(df.propnum.unique()):
            # print('table', table_name, 'scenario', scenario, 'property', p, i+1, 'of', num_prop)
            query = 'update ' + table_name + ' set '
            for column in df[df.propnum == p].variable.unique():
                value = df[(df.propnum == p) & (df.variable == column)].value.values[0]
                query = query + column + ' = ' + value + ', '
            query = query[:-2]
            if table_name == 'properties':
                query = query + ' where propnum = \'' + p + '\''
            else:
                query = query + ' where idp = \'' + p + '\''
            query = query + ' and scenario = \'' + scenario + '\''
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            cursor.close()

def load_probabilities(branch, sim=0):
    num_prop = len(branch.properties.propnum)
    uncertainty = {
        'idp': branch.properties.propnum.values,
        'drill_time': [1.0] * num_prop,
        'performance': [1.0] * num_prop,
        'profile': [1.0] * num_prop,
        'drill_cost': [1.0] * num_prop,
        'complete_cost': [1.0] * num_prop,
        'gas_price': [1.0] * num_prop,
        'oil_price': [1.0] * num_prop,
        'ngl_price': [1.0] * num_prop,
        'ngl_yield': [1.0] * num_prop,
        'btu': [1.0] * num_prop,
        'shrink': [1.0] * num_prop,
        'doe': [1.0] * num_prop,
        'gtp': [1.0] * num_prop,
        'total_capex': [1.0] * num_prop,
        'infra_cost': [1.0] * num_prop
    }
    uncertainty = pd.DataFrame(uncertainty)

    risk = {
        'idp': branch.properties.propnum.values,
        'drill_time': [1.0] * num_prop,
        'performance': [None] * num_prop,
        'in_zone': [None] * num_prop,
        'wellbore': [None] * num_prop,
        'profile': [1.0] * num_prop,
        'curtailment': [None] * num_prop,
        'frac_hit': [None] * num_prop,
        'spacing': [None] * num_prop,
        'drill_cost': [None] * num_prop,
        'complete_cost': [None] * num_prop,
        'abandon': [None] * num_prop,
        'downtime': [None] * num_prop,
        'frequency': [None] * num_prop,
        'duration': [None] * num_prop,
        'downtime_mult': [None] * num_prop,
        'downtime_cost': [None] * num_prop,
        'gas_downtime': [None] * num_prop,
        'oil_downtime': [None] * num_prop,
        'delay': [None] * num_prop,
    }
    risk = pd.DataFrame(risk)
    if branch.scenario.probability is not None:
        print('initializing probability model')
        sys.stdout.flush()
        df = load_probability_scenario(branch)
        if len(df) == 0:
            do_not_proceed = True
            while do_not_proceed is True:
                print('NO PROBABILITY DATA LOADED FOR \'' + branch.scenario.probability + '\'')
                cmd = input('proceed with run? (y/n): ')
                if cmd not in ('yes', 'no', 'YES', 'NO', 'y', 'n', 'Y', 'N'):
                    print('invalid command')
                else:
                    if cmd in ('no', 'NO', 'n', 'N'):
                        return
                    else:
                        do_not_proceed = False

        u = df[df.category == 'uncertainty']
        if len(u) > 0:
            for p in u.property_id.unique():
                col = u[u.property_id == p].property.values[0]
                properties = branch.properties[branch.properties[col] == p].propnum
                u_p = u[u.property_id == p]
                for t in u_p.type.unique():
                    d = prob_dict(u_p[u_p.type == t].value.values[0])
                    a = apply_uncertainty(d)
                    uncertainty.loc[uncertainty.idp.isin(properties), t] = a


        r = df[df.category == 'risk']
        abandon = None
        if len(r) > 0:
            for p in r.property_id.unique():
                col = r[r.property_id == p].property.values[0]
                properties = branch.properties[branch.properties[col] == p].propnum
                r_p = r[r.property_id == p]
                if len(properties) > 0:
                    if 'abandon' in r_p.type.values:
                        d = prob_dict(r_p[r_p.type == 'abandon'].value.values[0])
                        abandon = apply_risk(d)
                        if abandon is not None:
                            prop = np.random.choice(properties, 1)
                            rem_prop = [p for p in properties if p not in prop]
                            risk.loc[risk.idp.isin(prop), 'performance'] = 0.0
                            risk.loc[risk.idp.isin(prop), 'profile'] = 0.0
                            risk.loc[risk.idp.isin(prop), 'abandon'] = abandon
                            risk.loc[risk.idp.isin(rem_prop), 'abandon'] = None
                        else:
                            risk.loc[risk.idp.isin(properties), 'abandon'] = abandon

                    for t in r_p.type.values:
                        if t == 'abandon':
                            continue
                        if abandon is None:
                            if t == 'downtime':
                                d = prob_dict(r_p[r_p.type == t].value.values[0])
                                risk.loc[risk.idp.isin(properties), 'downtime'] = True
                                risk.loc[risk.idp.isin(properties), 'frequency'] = d['frequency']
                                risk.loc[risk.idp.isin(properties), 'duration'] = d['duration']
                                risk.loc[risk.idp.isin(properties), 'downtime_mult'] = d['mult']
                                risk.loc[risk.idp.isin(properties), 'downtime_cost'] = d['cost']
                            else:
                                d = prob_dict(r_p[r_p.type == t].value.values[0])
                                r_val = apply_risk(d)
                                if t == 'delay':
                                    if r_val is not None:
                                        r_val = int(r_val)
                                risk.loc[risk.idp.isin(properties), t] = r_val
                        else:
                            if t == 'downtime':
                                d = prob_dict(r_p[r_p.type == t].value.values[0])
                                risk.loc[risk.idp.isin(properties), 'downtime'] = True
                                risk.loc[risk.idp.isin(properties), 'frequency'] = d['frequency']
                                risk.loc[risk.idp.isin(properties), 'duration'] = d['duration']
                                risk.loc[risk.idp.isin(properties), 'downtime_mult'] = d['mult']
                                risk.loc[risk.idp.isin(properties), 'downtime_cost'] = d['cost']
                            else:
                                d = prob_dict(r_p[r_p.type == t].value.values[0])
                                r_val = apply_risk(d)
                                if t == 'delay':
                                    if r_val is not None:
                                        r_val = int(r_val)
                                risk.loc[risk.idp.isin(rem_prop), t] = r_val            

    uncertainty = uncertainty
    risk = risk

    save_probability_log(branch, uncertainty, risk, sim)

    return risk, uncertainty

def save_probability_log(branch, uncertainty, risk, sim):
    eng = engine(branch.tree.connection_dict) 
    u = uncertainty.melt(id_vars=['idp'], var_name='type', value_name='value')
    u['category'] = 'uncertainty'
    u['scenario'] = branch.scenario.scenario
    u['run_date'] = pd.Timestamp(branch.tree.run_time)
    u['simulation'] = sim
    u.to_sql(name='probability_log', con=eng,
             if_exists='append', method='multi',
             index=False, chunksize=500)
    r = risk.melt(id_vars=['idp'], var_name='type', value_name='value')
    r['category'] = 'risk'
    r['scenario'] = branch.scenario.scenario
    r['run_date'] = pd.Timestamp(branch.tree.run_time)
    r['simulation'] = sim
    r.to_sql(name='probability_log', con=eng,
             if_exists='append', method='multi',
             index=False, chunksize=500)
    return

def save_probability_output(framework, results):
    conn = connect(framework.branch.tree.connection_dict)
    eng = engine(framework.branch.tree.connection_dict)
    placeholders = ', '.join('?' * len(results.columns))
    query = str('insert into probability_monthly_output values (' + placeholders + ')')
    results.loc[:, 'prod_date'] = results.loc[:, 'prod_date'].dt.strftime('%')
    results.loc[:, 'run_date'] = results.loc[:, 'run_date'].dt.strftime('%')
    cursor = conn.cursor()
    cursor.executemany(query, list(results.itertuples(index=False, name=None)))
    conn.commit()
    cursor.close()
    return