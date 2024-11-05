import datetime

# focus regions where most of science was performed with start and end times
inset_map_settings = {
                    '20240528':{'start': datetime.datetime(2024, 5, 28, 13, 50),
                                'end'  : datetime.datetime(2024, 5, 28, 17, 10),
                                'extent':[-50, -40, 85, 86]
                               },
                    '20240530':{'start': datetime.datetime(2024, 5, 30, 12, 25),
                                'end'  : datetime.datetime(2024, 5, 30, 16, 30),
                                'extent':[-70, -50, 82.5, 86]
                               },
                    '20240531':{'start': datetime.datetime(2024, 5, 31, 13, 50),
                                'end'  : datetime.datetime(2024, 5, 31, 17, 10),
                                'extent': [-85, -50, 83, 85]
                               },
                    '20240603':{'start': datetime.datetime(2024, 6, 3, 13, 10),
                                'end'  : datetime.datetime(2024, 6, 3, 16, 0),
                                'extent':[-50, -40, 83, 85]
                               },
                    '20240605':{'start': datetime.datetime(2024, 6, 5, 12, 30),
                                'end'  : datetime.datetime(2024, 6, 5, 17, 0),
                                'extent':[-70, -40, 82.75, 84.25]
                               },
                    '20240606':{'start': datetime.datetime(2024, 6, 6, 13, 5),
                                'end'  : datetime.datetime(2024, 6, 6, 15, 30),
                                'extent':[-20, -5, 82.5, 84.15]
                               },
                    '20240607':{'start': datetime.datetime(2024, 6, 7, 14, 45),
                                'end'  : datetime.datetime(2024, 6, 7, 17, 20),
                                'extent':[-60, -40, 83, 85.25]
                               },
                    '20240610':{'start': datetime.datetime(2024, 6, 10, 11, 30),
                                'end'  : datetime.datetime(2024, 6, 10, 15, 30),
                                'extent':[-80, -70, 72.5, 77.5]
                               },
                    '20240611':{'start': datetime.datetime(2024, 6, 11, 12, 40),
                                'end'  : datetime.datetime(2024, 6, 11, 17, 5),
                                'extent':[-75, -45, 83, 86]
                               },
                    '20240613':{'start': datetime.datetime(2024, 6, 13, 12, 45),
                                'end'  : datetime.datetime(2024, 6, 13, 15, 25),
                                'extent':[-35, -10, 81.5, 85.25]
                               },
                    '20240725':{'start': datetime.datetime(2024, 7, 25, 12, 44),
                                'end'  : datetime.datetime(2024, 7, 25, 16, 4),
                                'extent':[-85, -48, 83, 85.5]
                               },
                    '20240729':{'start': datetime.datetime(2024, 7, 29, 13, 0),
                                'end'  : datetime.datetime(2024, 7, 29, 17, 5),
                                'extent': [-60, -40, 83, 85.5]
                               },
                    '20240730':{'start': datetime.datetime(2024, 7, 30, 12, 30),
                                'end'  : datetime.datetime(2024, 7, 25, 16, 10),
                                'extent':[-60, -20, 83, 85]
                               },
                    '20240801':{'start': datetime.datetime(2024, 8, 1, 13, 0),
                                'end'  : datetime.datetime(2024, 8, 1, 16, 15),
                                'extent':[-35, -10, 81, 85]
                               },
                    '20240802':{'start': datetime.datetime(2024, 8, 2, 12, 40),
                                'end'  : datetime.datetime(2024, 8, 2, 16, 40),
                                'extent':[-75, -45, 83, 85.5]
                               },
                    '20240807':{'start': datetime.datetime(2024, 8, 7, 12, 45),
                                'end'  : datetime.datetime(2024, 8, 7, 17, 0),
                                'extent':[-105, -90, 81.5, 84.25]
                               },
                    '20240808':{'start': datetime.datetime(2024, 8, 8, 12, 20),
                                'end'  : datetime.datetime(2024, 8, 8, 16, 40),
                                'extent':[-60, -35, 82, 84.75]
                               },
                    '20240809':{'start': datetime.datetime(2024, 8, 9, 12, 25),
                                'end'  : datetime.datetime(2024, 8, 9, 17, 10),
                                'extent':[-107, -90, 77.75, 81]
                               },
                    '20240815':{'start': datetime.datetime(2024, 8, 15, 13, 0),
                                'end'  : datetime.datetime(2024, 8, 15, 15, 50),
                                'extent':[-35, -10, 84, 85.5]
                               },
                   }

# dict of science flights to dates
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

# reverse the dict above so that they are dates:sfxy
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
