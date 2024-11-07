import datetime


# focus regions where most of science was performed with start and end times
inset_map_settings = {
                    '20240528':{'start': datetime.datetime(2024, 5, 28, 15, 5),
                                'end'  : datetime.datetime(2024, 5, 28, 15, 45),
                                'extent':[-55, -35, 84.75, 86.25],
                                'connectors': [1, 3]
                               },
                    '20240530':{'start': datetime.datetime(2024, 5, 30, 12, 25),
                                'end'  : datetime.datetime(2024, 5, 30, 16, 30),
                                'extent':[-70, -40, 82.5, 86],
                                'connectors': [0, 3]
                               },
                    '20240531':{'start': datetime.datetime(2024, 5, 31, 13, 50),
                                'end'  : datetime.datetime(2024, 5, 31, 17, 10),
                                'extent': [-85, -50, 83, 85],
                                'connectors': [0, 3]
                               },
                    '20240603':{'start': datetime.datetime(2024, 6, 3, 13, 10),
                                'end'  : datetime.datetime(2024, 6, 3, 16, 0),
                                'extent':[-65, -35, 82.75, 85.5],
                                'connectors': [0, 3]
                               },
                    '20240605':{'start': datetime.datetime(2024, 6, 5, 12, 30),
                                'end'  : datetime.datetime(2024, 6, 5, 17, 0),
                                'extent':[-70, -40, 82.75, 84.25],
                                'connectors': [0, 3]
                               },
                    '20240606':{'start': datetime.datetime(2024, 6, 6, 13, 5),
                                'end'  : datetime.datetime(2024, 6, 6, 15, 30),
                                'extent':[-20, -5, 82.5, 84.15],
                                'connectors': [0, 3]
                               },
                    '20240607':{'start': datetime.datetime(2024, 6, 7, 14, 45),
                                'end'  : datetime.datetime(2024, 6, 7, 17, 20),
                                'extent':[-65, -35, 82.5, 85.5],
                                'connectors': [0, 3]
                               },
                    '20240610':{'start': datetime.datetime(2024, 6, 10, 11, 30),
                                'end'  : datetime.datetime(2024, 6, 10, 15, 30),
                                'extent':[-80, -60, 72.5, 77.5],
                                'connectors': [0, 3]
                               },
                    '20240611':{'start': datetime.datetime(2024, 6, 11, 12, 40),
                                'end'  : datetime.datetime(2024, 6, 11, 17, 5),
                                'extent':[-85, -45, 83, 86],
                                'connectors': [0, 3]
                               },
                    '20240613':{'start': datetime.datetime(2024, 6, 13, 12, 45),
                                'end'  : datetime.datetime(2024, 6, 13, 15, 25),
                                'extent':[-45, -10, 81.5, 85.25],
                                'connectors': [0, 3]
                               },
                    '20240725':{'start': datetime.datetime(2024, 7, 25, 12, 44),
                                'end'  : datetime.datetime(2024, 7, 25, 16, 4),
                                'extent':[-85, -48, 83, 85.5],
                                'connectors': [0, 3]
                               },
                    '20240729':{'start': datetime.datetime(2024, 7, 29, 13, 0),
                                'end'  : datetime.datetime(2024, 7, 29, 17, 5),
                                'extent': [-60, -40, 83, 85.5],
                                'connectors': [0, 3]
                               },
                    '20240730':{'start': datetime.datetime(2024, 7, 30, 12, 30),
                                'end'  : datetime.datetime(2024, 7, 30, 16, 10),
                                'extent':[-60, -20, 83, 85],
                                'connectors': [0, 3]
                               },
                    '20240801':{'start': datetime.datetime(2024, 8, 1, 13, 0),
                                'end'  : datetime.datetime(2024, 8, 1, 16, 15),
                                'extent':[-35, -10, 81, 85],
                                'connectors': [0, 3]
                               },
                    '20240802':{'start': datetime.datetime(2024, 8, 2, 12, 40),
                                'end'  : datetime.datetime(2024, 8, 2, 16, 40),
                                'extent':[-75, -45, 83, 85.5],
                                'connectors': [0, 3]
                               },
                    '20240807':{'start': datetime.datetime(2024, 8, 7, 12, 45),
                                'end'  : datetime.datetime(2024, 8, 7, 17, 0),
                                'extent':[-105, -90, 81.5, 84.25],
                                'connectors': [0, 3]
                               },
                    '20240808':{'start': datetime.datetime(2024, 8, 8, 12, 20),
                                'end'  : datetime.datetime(2024, 8, 8, 16, 40),
                                'extent':[-70, -35, 82, 84.75],
                                'connectors': [0, 3]
                               },
                    '20240809':{'start': datetime.datetime(2024, 8, 9, 12, 25),
                                'end'  : datetime.datetime(2024, 8, 9, 17, 10),
                                'extent':[-110, -80, 79.5, 83.5],
                                'connectors': [0, 3]
                               },
                    '20240815':{'start': datetime.datetime(2024, 8, 15, 13, 0),
                                'end'  : datetime.datetime(2024, 8, 15, 15, 50),
                                'extent':[-35, 7, 83.75, 84.75],
                                'connectors': [0, 3]
                               },
                   }


flight_sf_to_date_dict = {'SF01': '20240528',
                     'SF02': '20240530',
                     'SF03': '20240531',
                     'SF04': '20240603',
                     'SF05': '20240605',
                     'SF06': '20240606',
                     'SF07': '20240607',
                     'SF08': '20240610',
                     'SF09': '20240611',
                     'SF10': '20240613',
                     'SF11': '20240725',
                     'SF12': '20240729',
                     'SF13': '20240730',
                     'SF14': '20240801',
                     'SF15': '20240802',
                     'SF16': '20240807',
                     'SF17': '20240808',
                     'SF18': '20240809',
                     'SF19': '20240815'}

flight_date_to_sf_dict = {'20240528': 'SF01',
                         '20240530': 'SF02',
                         '20240531': 'SF03',
                         '20240603': 'SF04',
                         '20240605': 'SF05',
                         '20240606': 'SF06',
                         '20240607': 'SF07',
                         '20240610': 'SF08',
                         '20240611': 'SF09',
                         '20240613': 'SF10',
                         '20240725': 'SF11',
                         '20240729': 'SF12',
                         '20240730': 'SF13',
                         '20240801': 'SF14',
                         '20240802': 'SF15',
                         '20240807': 'SF16',
                         '20240808': 'SF17',
                         '20240809': 'SF18',
                         '20240815': 'SF19'}


text_bg_colors = {'20240528': '#6c56f7',
                 '20240530': '#5691f7',
                 '20240531': '#56cef7',
                 '20240603': '#c8bcff',
                 '20240605': '#bcc9ff',
                 '20240606': '#bcd7ff',
                 '20240607': '#bce4ff',
                 '20240610': '#bcf3ff',
                 '20240611': '#bcfeff',
                 '20240613': '#bcfff5',
                 '20240725': '#fcf3bc',
                 '20240729': '#fce5bc',
                 '20240730': '#fcd8bc',
                 '20240801': '#fccfbc',
                 '20240802': '#fcc6bc',
                 '20240807': '#fcbcbc',
                 '20240808': '#fe7676',
                 '20240809': '#fd5050',
                 '20240815': '#fc2424'}
