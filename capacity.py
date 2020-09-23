from .utils import *
import time
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta
from pylatex import Document, Section,Center, Subsection,Tabu, LongTabu, Tabular, MiniPage, LineBreak, VerticalSpace, Math,NewLine, TikZ, Axis,Plot, Figure, Matrix, Alignat,\
     MultiColumn,MultiRow,PageStyle, Head, Foot,StandAloneGraphic,LargeText, MediumText,LineBreak, NewPage, Tabularx, TextColor,Command,basic,HugeText,\
     Subsubsection,LargeText,Hyperref,TextBlock, HorizontalSpace,SmallText,Itemize,Command
from pylatex.headfoot import simple_page_number
from pandas.plotting import register_matplotlib_converters
from pylatex.utils import italic, bold,NoEscape,verbatim
import textwrap
register_matplotlib_converters()

class Capacity():
    def __init__(self, branch):
        self.branch = branch
        self.date_range = self.branch.framework.date_range
        self.load_volumes()
        self.models = {}

    def load_volumes(self):
        print('loading capacity volumes')
        self.input_volumes = load_input_capacity_volumes(self)
        self.max_volumes = load_max_system_volumes(self)
        self.systems = load_system_output(self)
        self.systems.rename(columns={'propnum':'idp'}, inplace=True)
        self.system_list = list(self.systems.system.unique())
        for k, v in self.branch.model.items():
            v.system = self.systems[self.systems.idp == k].system.values[0]
        self.branch.framework.output = pd.merge(left=self.branch.framework.output,
                                                right=self.systems.drop(columns=['short_pad', 'pad']),
                                                how='inner', on=['idp'])

    def build_dataframe(self):
        print('building capacity dataframe')
        o = self.branch.framework.output
        df = o.groupby(by=['system', 'budget_type', 'pad', 'prod_date'], as_index=False).gross_gas.sum()
        self.system_dict = {}

        min_date = self.max_volumes.prod_date.min()
        max_date = self.max_volumes.prod_date.max()
        min_volumes = self.max_volumes[self.max_volumes.prod_date == min_date]
        max_volumes = self.max_volumes[self.max_volumes.prod_date == max_date]
        systems = self.max_volumes.system.unique()
        scenario = self.branch.scenario.capacity

        if min_date > self.date_range.min():
            prepend_dates = pd.date_range(self.date_range.min(), min_date - relativedelta(days=1))
            prepend = pd.DataFrame({'scenario': [scenario] * len(systems) * len(prepend_dates),
                                    'prod_date': list(prepend_dates) * len(systems),
                                    'system': list(systems) * len(prepend_dates),
                                    'max_volume': np.zeros(len(systems) * len(prepend_dates))})
            for sys in systems:
                prepend.loc[prepend.system == sys, 'max_volume'] = min_volumes.loc[min_volumes.system == sys, 'max_volume'].values
            self.max_volumes = pd.concat([prepend, self.max_volumes])

        if max_date < self.date_range.max():
            append_dates = pd.date_range(max_date + relativedelta(days=1), self.date_range.max())
            append = pd.DataFrame({'scenario': [scenario] * len(systems) * len(append_dates),
                                   'prod_date': list(append_dates) * len(systems),
                                   'system': list(systems) * len(append_dates),
                                   'max_volume': np.zeros(len(systems) * len(append_dates))})
            for sys in systems:
               append.loc[append.system == sys, 'max_volume'] = max_volumes.loc[max_volumes.system == sys, 'max_volume'].values
            self.max_volumes = pd.concat([self.max_volumes, append])

        for system in self.system_list:
            if system is None:
                continue
            sys = df.loc[df.system == system, :].copy()            
            tmp = pd.DataFrame(columns=['system', 'prod_date', 'base_volume', 'wedge_volume'])
            tmp['prod_date'] = self.date_range
            tmp['system'] = system
            bud = sys.groupby(by=['prod_date', 'budget_type'], as_index=False).gross_gas.sum()
            if bud[bud.budget_type == 'base'].empty:
                tmp['base_volume'] = 0.0
            else:
                tmp['base_volume'] = bud[bud.budget_type == 'base'].gross_gas.values
            if bud[bud.budget_type == 'wedge'].empty:
                tmp['wedge_volume'] = 0.0
            else:
                tmp['wedge_volume'] = bud[bud.budget_type == 'wedge'].gross_gas.values
            pad_volumes = sys.groupby(by=['prod_date', 'pad'], as_index=False).gross_gas.sum()
            wedge_pads = sys[sys.budget_type == 'wedge'].pad.unique()
            for p in wedge_pads:
                if pad_volumes[pad_volumes.pad == p].empty:
                    continue
                elif system not in ('LOW_PRESSURE', 'MEDIUM_PRESSURE'):
                    tmp.loc[:, p] = pad_volumes.loc[pad_volumes.pad == p, 'gross_gas'].values
            tmp = pd.merge(left=tmp, right=self.max_volumes.drop(columns=['scenario']),
                           how='left', on=['system', 'prod_date'])
            tmp = pd.merge(left=tmp, right=self.input_volumes.drop(columns=['scenario']),
                          how='left', on=['system', 'prod_date'])            
            self.system_dict[system] = tmp
        for system in self.system_list:
            if system not in ('LOW_PRESSURE', 'MEDIUM_PRESSURE'):
                continue
            sys = df.loc[df.system == system, :].copy()  
            pad_volumes = sys.groupby(by=['prod_date', 'pad'], as_index=False).gross_gas.sum()
            wedge_pads = sys[sys.budget_type == 'wedge'].pad.unique()
            for p in wedge_pads:
                if pad_volumes[pad_volumes.pad == p].empty:
                    continue
                else:
                    self.system_dict['STORY_GULCH'].loc[:, p] = pad_volumes.loc[pad_volumes.pad == p, 'gross_gas'].values

    def story_gulch(self):
        print('building story gulch model')
        sg = self.system_dict['STORY_GULCH']
        sg.drop(columns=['berry_volume', 'garden_gulch_volume'], inplace=True)
        lp = self.system_dict['LOW_PRESSURE'][['prod_date', 'base_volume', 'wedge_volume']]
        mp = self.system_dict['MEDIUM_PRESSURE'][['prod_date', 'base_volume', 'wedge_volume']]
        sg['lp_base'] = lp.base_volume.values
        sg['lp_wedge'] = lp.wedge_volume.values
        sg['mp_base'] = mp.base_volume.values
        sg['mp_wedge'] = mp.wedge_volume.values
        sg['total_volume'] = (sg.base_volume + sg.wedge_volume + sg.gas_lift_volume +
                              sg.lp_base + sg.lp_wedge + sg.mp_base + sg.mp_wedge)
        sg['forecast'] = sg.total_volume
        sg['overflow'] = sg.total_volume - sg.max_volume
        sg['report_overflow'] = sg['overflow']
        sg.loc[sg.overflow < 0.0, 'overflow'] = 0.0
        sg.loc[sg.forecast > sg.max_volume, 'forecast'] = sg.max_volume
        sg.loc[:, 'run_time'] = self.branch.tree.run_time
        self.models['story_gulch'] = sg

    def offload(self):
        print('building offload model')
        off = self.system_dict['OFFLOAD']
        off.drop(columns=['berry_volume', 'garden_gulch_volume', 'gas_lift_volume'], inplace=True)
        off['sg_overflow'] = self.models['story_gulch'].overflow.values
        off['total_volume'] = off.base_volume + off.wedge_volume + off.sg_overflow
        off['forecast'] = off.total_volume
        off['overflow'] = off.total_volume - off.max_volume
        off['report_overflow'] = off['overflow']
        off.loc[off.overflow < 0.0, 'overflow'] = 0.0
        off.loc[off.forecast > off.max_volume, 'forecast'] = off.max_volume
        off.loc[:, 'run_time'] = self.branch.tree.run_time
        self.models['offload'] = off

    def middle_fork(self):
        print('building middle fork model')
        mf = self.system_dict['MIDDLE_FORK']
        mf.fillna(0.0, inplace=True)
        mf['offload_volume'] = self.models['offload'].total_volume.values
        mf['total_volume'] = (mf.base_volume + mf.wedge_volume + mf.gas_lift_volume +
                              mf.berry_volume + mf.garden_gulch_volume + mf.offload_volume)
        mf['forecast'] = mf.total_volume
        mf['overflow'] = mf.total_volume - mf.max_volume
        mf['report_overflow'] = mf['overflow']
        mf.loc[mf.overflow < 0.0, 'overflow'] = 0.0
        mf.loc[mf.forecast > mf.max_volume, 'forecast'] = mf.max_volume
        mf.loc[:, 'run_time'] = self.branch.tree.run_time
        self.models['middle_fork'] = mf

    def pdf_report(self, file_path):
        self.report = Report(self)
        self.report.header()
        self.report.story_gulch()
        self.report.offload()
        self.report.middle_fork()
        if file_path == 'main':
            self.report.generate_pdf(str(self.branch.tree.name + '\\' +
                                         self.branch.scenario['scenario']),
                                     clean_tex=False, compiler='pdflatex')
        if file_path == 'archive':
            self.report.generate_pdf(str(self.branch.tree.name + '\\archive\\' +
                                         self.branch.scenario['scenario'] +
                                         '_capacity_model_' + self.branch.tree.run_time),
                                     clean_tex=False, compiler='pdflatex')


class Report(Document):
    def __init__(self, capacity):
        geometry_options = {
                            'head': '10pt',
                            'margin': '0.75in',
                            'bottom': '0.6in',
                            'includeheadfoot': False}
        super().__init__(geometry_options = geometry_options)
        self.capacity = capacity
        self.month1 = 4
        self.month2 = 5
        self.month3 = 6
        self.value1 = 3
        self.value2 = 4
        self.value3 = 5
        self.x = self.capacity.branch.framework.date_range

    def header(self):
        head_of_page = PageStyle('header', header_thickness=0.5, footer_thickness=0.0)
        with head_of_page.create(Head('L')):
            head_of_page.append(self.capacity.branch.scenario['scenario'])
            self.preamble.append(head_of_page)
            self.change_document_style('header')
        with head_of_page.create(Head('C')):
            head_of_page.append('Caerus Operating')
            self.preamble.append(head_of_page)
            self.change_document_style('header')
        with head_of_page.create(Head('R')):
            head_of_page.append(simple_page_number())
            self.preamble.append(head_of_page)
            self.change_document_style('header')

    def story_gulch(self):
        with self.create(Section('Story Gulch'))as section_sg:
            df = self.capacity.models['story_gulch']
            summary_table_sg = Tabular('ccc|ccc|ccc', row_height=.40, width=9, booktabs=True)
            avg_tablemaker(summary_table_sg, df, self.month1, self.month2, self.month3)
            self.append(summary_table_sg)            
            with self.create(Figure(position='h!')) as sg_plot:
                sg_fig,(ax1) = plt.subplots(nrows=1, ncols=1, sharex=True,
                                            sharey=False, squeeze=True, figsize=(14,4))
                self.append(NoEscape(r''))
                ax1.stackplot(self.x, df.base_volume+df.wedge_volume, df.gas_lift_volume,
                              df.lp_base+df.lp_wedge, df.mp_base+df.mp_wedge,
                              labels=['Base', 'Gas Lift', 'Low Pressure', 'Medium Pressure'],
                              colors=['blue', 'gold', 'cornflowerblue', 'goldenrod'])
                ax1.plot(self.x, df.max_volume, label='Capacity')
                ax1.legend(loc='upper left')
                ax1.tick_params(labelsize=14)
                sg_plot.add_plot(width=NoEscape(r'1\textwidth'))
                table_sg = Tabular('ccc|ccc|ccc', row_height=.10, width=9, booktabs=True)
                table_sg.add_row('Date', 'Gas, MCF', 'Delta',
                                 'Date', 'Gas, MCF', 'Delta',
                                 'Date', 'Gas, MCF', 'Delta')
                table_sg.add_hline()
                tablemaker(table_sg,
                                 df.prod_date.astype(str).tolist(),
                                 df.forecast.values.tolist(),
                                 df.report_overflow.values.tolist())
            section_sg.append(table_sg)
        self.append(NewPage())

    def offload(self):
        with self.create(Section('Offload'))as section_off:
            df = self.capacity.models['offload']
            summary_table_off = Tabular('ccc|ccc|ccc', row_height=.40, width=9, booktabs=True)
            avg_tablemaker(summary_table_off, df, self.month1, self.month2, self.month3)
            self.append(summary_table_off)            
            with self.create(Figure(position='h!')) as off_plot:
                off_fig,(ax1) = plt.subplots(nrows=1, ncols=1, sharex=True,
                                            sharey=False, squeeze=True, figsize=(14,4))
                self.append(NoEscape(r''))
                ax1.stackplot(self.x, df.base_volume, df.wedge_volume, df.sg_overflow,
                              labels=['Base', 'Wedge', 'Medium Pressure'],
                              colors=['gold', 'goldenrod', 'cornflowerblue'])
                ax1.plot(self.x, df.max_volume, label='Capacity')
                ax1.legend(loc='upper left')
                ax1.tick_params(labelsize=14)
                off_plot.add_plot(width=NoEscape(r'1\textwidth'))
                table_off = Tabular('ccc|ccc|ccc', row_height=.10, width=9, booktabs=True)
                table_off.add_row('Date', 'Gas, MCF', 'Delta',
                                 'Date', 'Gas, MCF', 'Delta',
                                 'Date', 'Gas, MCF', 'Delta')
                table_off.add_hline()
                tablemaker(table_off,
                                 df.prod_date.astype(str).tolist(),
                                 df.forecast.values.tolist(),
                                 df.report_overflow.values.tolist())
            section_off.append(table_off)
        self.append(NewPage())                              

    def middle_fork(self):
        with self.create(Section('Middle Fork'))as section_mf:
            df = self.capacity.models['middle_fork']
            summary_table_mf = Tabular('ccc|ccc|ccc', row_height=.40, width=9, booktabs=True)
            avg_tablemaker(summary_table_mf, df, self.month1, self.month2, self.month3)
            self.append(summary_table_mf)            
            with self.create(Figure(position='h!')) as mf_plot:
                mf_fig,(ax1) = plt.subplots(nrows=1, ncols=1, sharex=True,
                                            sharey=False, squeeze=True, figsize=(14,4))
                self.append(NoEscape(r''))
                ax1.stackplot(self.x, df.base_volume, df.wedge_volume, df.offload_volume,
                              df.gas_lift_volume, df.berry_volume, df.garden_gulch_volume,
                              labels=['Base', 'Wedge', 'Offload', 'Gas Lift', 'Berry', 'Garden Gulch'],
                              colors=['blue', 'cornflowerblue', 'orange', 'gold', 'goldenrod', 'red'])
                ax1.plot(self.x, df.max_volume, label='Capacity')
                ax1.legend(loc='upper left')
                ax1.tick_params(labelsize=14)
                mf_plot.add_plot(width=NoEscape(r'1\textwidth'))
                table_mf = Tabular('ccc|ccc|ccc', row_height=.10, width=9, booktabs=True)
                table_mf.add_row('Date', 'Gas, MCF', 'Delta',
                                 'Date', 'Gas, MCF', 'Delta',
                                 'Date', 'Gas, MCF', 'Delta')
                table_mf.add_hline()
                tablemaker(table_mf,
                                 df.prod_date.astype(str).tolist(),
                                 df.forecast.values.tolist(),
                                 df.report_overflow.values.tolist())
            section_mf.append(table_mf)
        self.append(NewPage())
    
