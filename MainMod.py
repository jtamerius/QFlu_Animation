import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
from datetime import datetime as dt
from datetime import timedelta
import glob
import os
import shutil

def ProcessData():

    FullMx = pd.DataFrame(
        columns=['SofiaSN', 'TestDate', 'Facility#', 'City', 'State', 'Zip Code', 'PatientAge', 'FluA', 'FluB',
                 'Overall Result', 'County', 'Facility Type'])

    def Read_Qxls(fin):
        tmp = pd.read_excel(fin)
        Y = {'Facility Type': ['FacilityDescription', 'FacilityType', 'Facility Type'],
             'Overall Result': ['OverallResult', 'Overall Result', '''Overall Result (DP's = Neg)'''],
             'Zip Code': ['Zip Code', 'ZipCode', 'Zip'],
             'SofiaSN': ['SofiaSerNum', 'Sofia Ser Num', 'SofiaSN'],
             'Facility#': ['Facility#', 'Facility #', 'Facility Num', 'Facility'], 'City': ['City'],
             'TestDate': ['TestDate', 'Date', 'Test Date'], 'FluA': ['Flu A', 'FluA', 'Result1'],
             'FluB': ['Flu B', 'FluB', 'Result2'],
             'County': ['County'], 'PatientAge': ['PatientAge', 'Age'], 'State': ['State', 'ST']}
        new_columns = []
        for hdr in list(tmp.columns.values):
            new_columns.append(search(Y, hdr))
        tmp.columns = new_columns
        return tmp

    def search(keys, searchFor):
        for k in keys:
            if searchFor in keys[k]:
                return k

    print('Reading Files...')
    for fin in glob.glob('/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Data/RawData/*.xlsx'):
        print(fin)
        FullMx = pd.concat([Read_Qxls(fin), FullMx])

    FullMx['Zip Code'] = FullMx['Zip Code'].astype('str').str[:5]
    FullMx.to_pickle('/Users/jamestamerius/Dropbox/Documents/Projects/Qflu2017_2018season/Qflu_raw')

    ############# REMOVE DATA WITH POSITVE FLU_A AND FLU_B ####################
    print('Filtering Data...')
    ins = ~ ((FullMx['FluA'] == 'positive') & (FullMx['FluB'] == 'positive'))
    FullMx = FullMx[ins]

    ############# GROUP POSITIVE COUNTS ########################################
    print('Grouping/Counting Data...')

    def _ct_id_pos(grp):
        return grp['SofiaSN'].iloc[0], grp['Zip Code'].iloc[0], grp['State'].iloc[0], grp[grp.FluA == 'positive'].shape[
            0], grp[grp.FluB == 'positive'].shape[0], grp.shape[0]

    FullMx_prime = FullMx.groupby(['TestDate', 'SofiaSN']).apply(_ct_id_pos).reset_index()
    FullMx_prime[['SofiaSN', 'Zip Code', 'ST', 'Pos_A', 'Pos_B', 'Total_Tests']] = FullMx_prime[0].apply(pd.Series)
    FullMx_prime.drop([0], axis=1, inplace=True)

    ############# ZIPCODE -- LAT/LON LOOKUP ##########################################
    print('Matching Lat/Lon with Zip Codes...')
    Zip2LatLon = pd.read_csv(
        '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Data/ZipCode2LatLon.csv')
    Zip2LatLon.columns = ['Zip Code', 'Lat', 'Lon']
    FullMx_prime['Zip Code'] = FullMx_prime['Zip Code'].astype(float).astype(int)

    QFlu = pd.DataFrame()
    for name, group in FullMx_prime.groupby(['TestDate']):
        tmpMx = pd.merge(group, Zip2LatLon, on='Zip Code', how='inner')
        QFlu = QFlu.append(tmpMx)

    ########### Aggregate by zip code
    print('Aggregating data...')

    def sum_by_zip(grp):
        return grp['TestDate'].iloc[0], grp['ST'].iloc[0], grp['Zip Code'].iloc[0], sum(grp.Pos_A), sum(grp.Pos_B), \
               sum(grp.Total_Tests), grp['Lat'].iloc[0], grp['Lon'].iloc[0]

    QFlu_prime = QFlu.groupby(['Zip Code', 'TestDate']).apply(sum_by_zip).reset_index()
    QFlu_prime[['TestDate', 'ST', 'Zip Code', 'Pos_A', 'Pos_B', 'Total_Tests', 'Lat', 'Lon']] = QFlu_prime[0].apply(
        pd.Series)
    QFlu_prime.drop([0], axis=1, inplace=True)
    del QFlu

    ############# Make new variables (A to B ratio) and total positive
    QFlu_prime['A_to_B'] = QFlu_prime.Pos_A / (QFlu_prime.Pos_A + QFlu_prime.Pos_B)
    QFlu_prime['A_to_B'] = QFlu_prime['A_to_B'].fillna(0)
    QFlu_prime['TotPos'] = QFlu_prime['Pos_A'] + QFlu_prime['Pos_B']
    QFlu_prime['TestDate'] = pd.to_datetime(QFlu_prime.TestDate)

    QFlu_prime.to_csv('/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Data/QFlu_processed_I.csv',
                      index=False)

    ###################### Smooth signal
    print('Smoothing signal...')

    def SmoothSig(grp):
        grp2 = pd.DataFrame()
        ST = grp.iloc[0]['ST']
        zip = grp.iloc[0]['Zip Code']
        lat = grp.iloc[0].Lat
        lon = grp.iloc[0].Lon
        idx = pd.date_range(pd.to_datetime('8/20/2015'), pd.to_datetime('today'))
        grp.index = pd.to_datetime(grp['TestDate'])
        grp = grp.reindex(idx, fill_value=0)
        grp.index = pd.DatetimeIndex(grp.index)

        grp2[['TotPos', 'A_to_B', 'Pos_A', 'Pos_B', 'Total_Tests']] = grp[
            ['TotPos', 'A_to_B', 'Pos_A', 'Pos_B', 'Total_Tests']].rolling(21, min_periods=1).mean()
        grp2.reset_index(drop=True, inplace=True)
        grp2['Lat'] = lat
        grp2['Lon'] = lon
        grp2['TestDate'] = idx
        grp2['Zip Code'] = zip
        grp2['ST'] = ST

        return grp2

    QFlu_processed = QFlu_prime.groupby('Zip Code').apply(SmoothSig)
    QFlu_processed = QFlu_processed[QFlu_processed['TotPos'] > 0]
    QFlu_processed.reset_index(inplace=True, drop=True)

    # save to csv
    print('Saving to csv...')
    QFlu_processed.to_csv(
        '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Data/QFlu_processed_II.csv', index=False)

def MkRunner(ST, date0, date1, abs_prc):
    # date is first date in yyyy-mm-dd
    date0 = pd.to_datetime(date0)
    date1 = pd.to_datetime(date1)
    fname = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Data/QFlu_processed_II.csv'
    df = pd.read_csv(fname)
    df.TestDate = pd.to_datetime(df.TestDate)
    if ST != 'US':
        df = df[df['ST'] == ST]

    df = df.groupby(['TestDate']).sum()
    df.reset_index(drop=False, inplace=True)
    df = df[(df.TestDate >= date0) & (df.TestDate <= date1)]
    idx = pd.date_range(date0, date1)
    df.index = df['TestDate']
    df = df.reindex(idx, fill_value=0)
    df.reset_index(drop=True, inplace=True)

    if abs_prc == 'abs':
        PosA = df['Pos_A'].rolling(21, min_periods=1).mean()
        PosB = df['Pos_B'].rolling(21, min_periods=1).mean()
    elif abs_prc == 'prc':
        PosA = (df['Pos_A'] / df['Total_Tests']).rolling(21, min_periods=1).mean()
        PosB = (df['Pos_B'] / df['Total_Tests']).rolling(21, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 1))
    ax.fill_between(np.arange(len(PosA)), 0, PosA, color='red', alpha='0.75', linewidth=.25)
    ax.fill_between(np.arange(len(PosA)), 0, PosB, color='blue', alpha='0.75', linewidth=.25)
    plt.axis('off')

    directory = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Figures/'+ST+'/Runners/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Figures/'+ST+'/Runners/Runner.png'
    fig.savefig(fname,transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
    img = Image.open(fname)

    #######Crop Image
    image_data = np.asarray(img)
    image_data_bw = image_data.mean(axis=2)
    non_empty_cols = np.where(image_data_bw.min(axis=0) < 191.25)[0]
    non_empty_rows = np.where(image_data_bw.min(axis=1) < 191.25)[0]
    cropBox = (min(non_empty_cols), min(non_empty_rows), max(non_empty_cols), max(non_empty_rows))
    img2 = img.crop(cropBox)

    #######Save Image
    img2.save(fname, "PNG")
    return max(max(PosA), max(PosB)), len(PosA)

def MkMap(ST,start_date,end_date):
    # create new figure, axes instances.
    fig = plt.figure(figsize=(11.81, 9.09), dpi=600)
    ax = fig.add_axes([0.001, 0.001, .999, .999])

    if ST == 'US':
        # setup map projection:  c, l, i, h, f
        m = Basemap(llcrnrlon=-130., llcrnrlat=10, urcrnrlon=-50, urcrnrlat=46, \
                    resolution='l', lon_0=-96, lat_0=40, projection='aea' \
                    , ellps='GRS80', area_thresh=10000, lat_1=20, lat_2=60)

        fname_ST = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/QGIS/Qflu_QGIS_shpfiles/cb_2016_us_state_500k/cb_2016_us_state_500k'
        m.readshapefile(fname_ST, 'states', drawbounds=False)
        patches = []
        for info, shape in zip(m.states, m.states):
            patches.append(Polygon(np.array(shape), True))
        ax.add_collection(PatchCollection(patches, facecolor='#dedede', edgecolor='w', linewidths=.5, zorder=0))

    elif ST == 'FL':
        # setup map projection:  c, l, i, h, f
        m = Basemap(llcrnrlon = -90, llcrnrlat = 23, urcrnrlon = -79, urcrnrlat = 32, \
                    resolution = 'l', lon_0 = -81, lat_0 = 24, projection = 'aea' \
                    ,ellps = 'GRS80', area_thresh = 10000)

    elif ST == 'TX':
        # setup map projection:  c, l, i, h, f
        m = Basemap(llcrnrlon = -107.75, llcrnrlat = 23.5, urcrnrlon = -91.5, urcrnrlat = 36.75, \
                    resolution = 'l', lon_0 = -100, lat_0 = 30.8, projection = 'aea' \
                    ,ellps = 'GRS80', area_thresh = 10000)

    elif ST == 'GA':
        # setup map projection:  c, l, i, h, f
        m = Basemap(llcrnrlon = -87.5, llcrnrlat = 29.25, urcrnrlon = -80.5, urcrnrlat = 35.25, \
                    resolution = 'l', lon_0 = -83.25, lat_0 = 32.5, projection = 'aea' \
                    ,ellps = 'GRS80', area_thresh = 10000)

    elif ST == 'MS':
        # setup map projection:  c, l, i, h, f
        m = Basemap(llcrnrlon=-94, llcrnrlat=28.75, urcrnrlon=-86, urcrnrlat=35.5, \
                    resolution='l', lon_0=-89.25, lat_0=32.5, projection='aea' \
                    , ellps='GRS80', area_thresh=10000)

    elif ST == 'AL':
        # setup map projection:  c, l, i, h, f
        m = Basemap(llcrnrlon=-90.5, llcrnrlat=28.75, urcrnrlon=-83, urcrnrlat=35.5, \
                    resolution='l', lon_0=-86.25, lat_0=32.5, projection='aea' \
                    , ellps='GRS80', area_thresh=10000)

    if ST != 'US':# Make urban areas background

        # Make US background
        fname_US = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Mapping/cb_2016_us_state_500k/cb_2016_us_state_500k'
        m.readshapefile(fname_US, 'states', drawbounds=False)
        patches_all_ST = []
        patches_ST = []
        for info, shape in zip(m.states_info, m.states):
            if info['STUSPS'] == ST:
                patches_ST.append(Polygon(np.array(shape), True))
            else:
                patches_all_ST.append(Polygon(np.array(shape), True))
        ax.add_collection(PatchCollection(patches_all_ST, facecolor='#C6C6C6', edgecolor='#F5F5F5', linewidths=.5, zorder=0))
        ax.add_collection(PatchCollection(patches_ST, facecolor='#EEEEEE', edgecolor='w', linewidths=2, zorder=0))

        fname_UA = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Mapping/cb_2016_us_ua10_500k/cb_2016_us_ua10_500k'
        m.readshapefile(fname_UA, 'UA', drawbounds=False)
        patches = []
        for info, shape in zip(m.UA, m.UA):
            patches.append(Polygon(np.array(shape), True))
        ax.add_collection(PatchCollection(patches, facecolor='#D7D7D7', edgecolor=None, linewidths=.5, zorder=1))

        # Plot roads
        fname_rds = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Mapping/MajorRoadsUS/MajorRoadsUS'
        m.readshapefile(fname_rds, 'roads', drawbounds=False)
        for info, shape in zip(m.roads, m.roads):
            xx, yy = zip(*shape)
            m.plot(xx, yy, color='#D7D7D7', linewidth=.5, zorder=0)


    ax.axis('off')

    # import Qflu data
    fname = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Data/QFlu_processed_II.csv'
    Qflu = pd.read_csv(fname)
    Qflu.TestDate = pd.to_datetime(Qflu.TestDate)
    Qflu = Qflu[(Qflu['TestDate'] > pd.to_datetime(start_date)) & (Qflu['TestDate'] <= pd.to_datetime(end_date))]
    Qflu = Qflu.sort_values('TestDate')

    # define the color map
    cmap = pd.read_csv('/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Resources/cmap.csv')
    cmap = cmap.as_matrix()
    cm = mpl.colors.ListedColormap(cmap / 255.0)

    # begin for loop
    cnt = 1

    directory = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Figures/'+ ST + '/Figs_I'
    directory2 = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Figures/'+ ST + '/Figs_II'

    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory2):
        os.makedirs(directory2)

    for day in Qflu['TestDate'].sort_values().unique():
        print(pd.to_datetime(day))
        d = '{0:04d}'.format(cnt)
        df = Qflu[Qflu['TestDate'] == pd.to_datetime(day)]
        df = df.sort_values('Zip Code')
        Lon = df['Lon'].as_matrix()
        Lat = df['Lat'].as_matrix()
        Total = df['TotPos'].as_matrix()
        AB = df['A_to_B'].as_matrix()
        points = m.scatter(Lon, Lat, latlon=True, s=Total * 15, cmap=cm, c=AB, alpha=0.8, marker="o", linewidth=.25)
        f = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Figures/'+ ST + '/Figs_I/Fig' + d + '.png'
        plt.savefig(f, dpi=600, bbox_inches='tight')
        points.remove()
        cnt = cnt + 1

def AddRunner2Map(ST,min_value, max_value, w_bar,start_day):
    # Get filenames
    filenames = sorted(glob.glob('/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Figures/'+ST+'/Figs_I/*.png'))
    print(filenames)

    # open images
    legend = Image.open('/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Legends/Legend_15.png').convert(
        "RGBA")

    rnnr_fname = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Figures/'+ST+'/Runners/Runner.png'

    barPlot = Image.open(rnnr_fname).convert("RGBA")

    # set size of barplot, resize
    h_bar = 450
    #w_bar = 5000
    barPlot = barPlot.resize((w_bar, h_bar))

    # start Date
    Day = dt.strptime(start_day, '%Y-%m-%d')

    # calculate/define increment
    inc = barPlot.size[0] / (len(filenames) - 1)
    wnd_sz = 21 * inc
    x_rec1 = 0

    # set size of overlay
    h_padding = 200
    w_padding = 500
    h_overlay = h_bar + h_padding
    w_overlay = w_bar + w_padding
    x_inset = 400
    y_inset = round(h_padding / 2)

    # fnt and text settings
    txtfill = (110, 110, 110, 255)
    txtfill2 = (75, 75, 75, 255)
    txtfill3 = txtfill2

    fnt = ImageFont.truetype("Library/Fonts/Arial.ttf", 100)
    fnt2 = ImageFont.truetype("Library/Fonts/Arial.ttf", 112)

    for filename in filenames:
        # load map
        source_img = Image.open(filename).convert("RGBA")
        w, h = source_img.size
        source_img = source_img.crop((0, 0, w-w%2, h-h%2))

        # size of map
        x_map = round(source_img.size[0])
        y_map = round(source_img.size[1])

        # create master overlay
        overlay_master = Image.new("RGBA", (w_overlay, h_overlay), color=(255, 255, 255, 0))
        overlay_org = ((round(x_map / 2 - (w_overlay / 2)), y_map - h_overlay - 100))
        source_img.paste(overlay_master, overlay_org, overlay_master)

        # insert legend
        w_leg, h_leg = legend.size
        legend_img = Image.new("RGBA", (w_leg + 100, h_leg + 100), color=(255, 255, 255, 255))
        legend_img.paste(legend, (50, 50), legend)
        source_img.paste(legend_img, (250, 250), legend_img)

        # draw on overlay
        dr1 = ImageDraw.Draw(source_img)

        # insert overlay on map
        source_img.paste(barPlot, (overlay_org[0] + x_inset, overlay_org[1] + y_inset), barPlot)

        # Draw grid lines and dyanmic overlay
        mx = max_value
        if mx == 0:
            mx = 1

        mn = min_value
        line1_val = round(int(mx)/100)*100
        line2_val = round(int(mx)/100)*50
        line1 = round(overlay_org[1] + h_bar - line1_val * (h_bar / mx) + y_inset)
        line2 = round(overlay_org[1] + h_bar - line2_val * (h_bar / mx) + y_inset)
        line3 = round(overlay_org[1] + h_bar - 7 + y_inset)
        x_0 = round(overlay_org[0] + x_inset)
        r_int = round(x_0 + x_rec1 + wnd_sz / 2)
        l_int = round(x_0 + x_rec1 - wnd_sz / 2)
        h_doverlay = overlay_org[1] + y_inset

        box_w = 8
        if (l_int > x_0) & (round(x_0 + x_rec1 - wnd_sz / 2 - x_0 - round(box_w / 2)) > 0) & (r_int <= x_0 + w_bar):
            dr1.rectangle([l_int - round(box_w / 2), line3, r_int + round(box_w / 2), line3 + 35], fill=txtfill3)
            dr1.line([x_0, line1, round(x_0 + x_rec1 - wnd_sz / 2), line1], fill=txtfill, width=box_w)
            dr1.line([x_0, line2, round(x_0 + x_rec1 - wnd_sz / 2), line2], fill=txtfill, width=box_w)
            dr1.line([l_int, line3, l_int, line1], fill=txtfill, width=box_w)  # left interval
            left_dover = Image.new("RGBA", (round(x_0 + x_rec1 - wnd_sz / 2 - x_0 - round(box_w / 2)), h_bar),
                                   color=(255, 255, 255, 0))
            source_img.paste(left_dover, (x_0, h_doverlay), left_dover)
        elif (r_int >= x_0 + w_bar):
            dr1.line([x_0, line1, round(x_0 + x_rec1 - wnd_sz / 2), line1], fill=txtfill, width=box_w)
            dr1.line([x_0, line2, round(x_0 + x_rec1 - wnd_sz / 2), line2], fill=txtfill, width=box_w)
            dr1.line([l_int, line3, l_int, line1], fill=txtfill, width=box_w)  # left interval
            left_dover = Image.new("RGBA", (round(x_0 + x_rec1 - wnd_sz / 2 - x_0 - round(box_w / 2)), h_bar),
                                   color=(255, 255, 255, 0))
            source_img.paste(left_dover, (x_0, h_doverlay), left_dover)
            dr1.rectangle([l_int - round(box_w / 2), line3, x_0 + w_bar, line3 + 35], fill=txtfill3)
        elif (r_int <= x_0 + w_bar):
            dr1.rectangle([x_0, line3, r_int, line3 + 35], fill=txtfill3)

        if r_int <= x_0 + w_bar:
            dr1.line([round(x_0 + x_rec1 + wnd_sz / 2), line1, x_0 + w_bar, line1], fill=txtfill,
                     width=box_w)  # top grid line
            dr1.line([round(x_0 + x_rec1 + wnd_sz / 2), line2, x_0 + w_bar, line2], fill=txtfill, width=box_w)
            dr1.line([r_int, line3, r_int, line1], fill=txtfill, width=box_w)  # right interval
            right_dover = Image.new("RGBA", (round(x_0 + w_bar - r_int), h_bar), color=(255, 255, 255, 0))
            source_img.paste(right_dover, (r_int + round(box_w / 2), h_doverlay), right_dover)
        else:
            dr1.rectangle([l_int, line3, x_0 + w_bar, line3 + 35], fill=txtfill3)

        # Add Date Text
        date = Day.strftime("%B %d, %Y")
        draw = ImageDraw.Draw(source_img)
        fntDate = ImageFont.truetype("Library/Fonts/Arial.ttf", 150)
        w, h = draw.textsize(date, font=fntDate)
        draw.text((round(overlay_org[0] + w_overlay / 2 - w / 2), overlay_org[1] - h), date, font=fntDate,
                  fill=txtfill2)

        # Draw y-ticks
        dr4 = ImageDraw.Draw(source_img)
        w, h = dr4.textsize(str(line1_val), font=fnt)
        dr4.text((round(overlay_org[0] + x_inset - w - 25), line1 - h / 2), str(line1_val), font=fnt, fill=txtfill)
        w, h = dr4.textsize(str(line2_val), font=fnt)
        dr4.text((overlay_org[0] + x_inset - w - 25, line2 - h / 2), str(line2_val), font=fnt, fill=txtfill)
        w, h = dr4.textsize('0', font=fnt)
        dr4.text((overlay_org[0] + x_inset - w - 25, line3 - h / 2), '0', font=fnt, fill=txtfill)

        # Y-axis label
        label = '# Positive'
        w, h = dr4.textsize(label, font=fnt2)
        im = Image.new("RGBA", (w, h))
        d = ImageDraw.Draw(im)
        d.text((0, 0), label, font=fnt2, fill=txtfill)
        im = im.rotate(90, expand=1)
        source_img.paste(im, (overlay_org[0] + 0, round(line2 - w / 2)), im)
        # source_img.show()

        # counters
        Day = Day + timedelta(days=1)  # add one day to date
        x_rec1 += inc

        # save image
        source_img = source_img.convert("RGB")
        filename = filename[:77] + 'Figs_II/Fig' + filename[-8:]
        source_img.save(filename, "png")

def MkVideo(ST, start_date, end_date, speed):
    directory = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Figures/'+ST+'/Figs_II/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    os.chdir(directory)

    directory2 = '/Users/jamestamerius/Dropbox/Documents/Projects/QuidelData/Python/Figures/' + ST + '/Video'

    if not os.path.exists(directory2):
        os.makedirs(directory2)


    mpeg_command = 'ffmpeg -framerate ' + str(speed) +' -f image2 -start_number 1 -i fig%4d.png -f mp4 -vf scale=-2:ih -vcodec libx264 -pix_fmt yuv420p ' + 'tmp.mp4'


    os.system(mpeg_command)
    #os.system("ffmpeg -framerate 18 -f image2 -start_number 1 -i fig%4d.png -f mp4 -vcodec libx264 -pix_fmt yuv420p tmp.mp4")
    # os.system('ffmpeg -i tmp.mp4 -r 25 -f mpeg -vcodec mpeg1video -ar 48000 -b:v 5000k -b:a 128k -ar 44100 -ac 1 -y OUTFILE.mpg')

    #os.system("ffmpeg -framerate 18  -start_number 1 -i fig%4d.png -b:v 2M -vcodec msmpeg4 tmp.wmv")

    # f1 = directory + 'tmp.mp4'
    # os.remove(f1)
    # f2 = directory2 + '/' + start_date[0:4] + '-' + end_date[0:4] + '_' + ST + '.mp4

    f1 = directory + 'tmp.mp4'
    f2 = directory2 + '/' + start_date[0:4] + '-' + end_date[0:4] + '_' + ST + '.mp4'
    shutil.move(f1, f2)






#ProcessData()


speed = 10 ###slow = 12, fast 25
states = ['FL','GA','TX']
start_date = '2017-8-1'
end_date = '2018-8-1'
for ST in states:
    [a, b] = MkRunner(ST, start_date, end_date, 'abs')
    MkMap(ST, start_date, end_date)
    AddRunner2Map(ST, 0, int(a), 3000, start_date)
    MkVideo(ST, start_date, end_date,speed)


# print('State: ' + ST)
# print('start_date: ' + start_date)
# print('end_date: ' + end_date)
#
# [a,b] = MkRunner(ST, start_date, end_date, 'abs')
# MkMap(ST,start_date, end_date)
# AddRunner2Map(ST,0, int(a), 3000, start_date)
