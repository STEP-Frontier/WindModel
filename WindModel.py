from collect_data import collect_sonde
import os
import argparse
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from analysis.probability_ellipse import ConfidenceEllipse

if __name__ == '__main__':

######################## USER SETTING VARIABLES ########################
    alt_sta = 100
    alt_end = 3000
    alt_del = 300
########################################################################


    this_file_path = os.path.abspath(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='ラジオゾンデ観測データをダウンロードするスクリプト')

    parser.add_argument('station_name')
    parser.add_argument('year_range')
    parser.add_argument('month_range')
    parser.add_argument('day_range')
    parser.add_argument('hour')

    parser.add_argument('--output_dir', default='wind_Rawin')

    args = parser.parse_args()

    station_name = args.station_name
    year_range  = [int(y) for y in args.year_range.split(':')]
    month_range = [int(m) for m in args.month_range.split(':')]
    day_range   = [int(d) for d in args.day_range.split(':')]
    hour = int(args.hour)

    years = collect_sonde.getRange(year_range)
    months = collect_sonde.getRange(month_range)
    days = collect_sonde.getRange(day_range)

    savedir = os.path.join(this_file_path, '../', args.output_dir, station_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    df_array = []
    alt_array = np.arange(alt_sta, alt_end+alt_del, alt_del)
    nalt = len(alt_array)
    wind_u_list = []
    wind_v_list = []
    speed_list = []
    theta_list = []

# collect wind data
    for y in years:
        for m in months:
            for d in days:
                date_str = '{:0>4}{:0>2}{:0>2}{:0>2}'.format(y, m, d, hour)
                csv_filename = os.path.join(savedir, 'Rawin_'+date_str+'.csv')
                df = collect_sonde.getSondeDataAsDataFrame(station_name, y, m, d, hour)
                alt_prev = df['GEO_HGT']
                theta = df['DIRECTION_DEGREE'] * np.pi/180.0
                speed = df['SPEED']
                wind_u = np.array(-speed*np.sin(theta))
                wind_v = np.array(-speed*np.cos(theta))

                wind_u_new = interpolate.interp1d(alt_prev, wind_u, kind='linear')
                wind_v_new = interpolate.interp1d(alt_prev, wind_v, kind='linear')
                speed_new  = interpolate.interp1d(alt_prev, speed , kind='linear')
                theta_new  = interpolate.interp1d(alt_prev, theta , kind='linear')

                wind_u_list.append(wind_u_new(alt_array))
                wind_v_list.append(wind_v_new(alt_array))
                speed_list.append( speed_new(alt_array))
                theta_list.append( theta_new(alt_array))
    
    # list型をndarray型に変換
    wind_u_array = np.array(wind_u_list)
    wind_v_array = np.array(wind_v_list)
    speed_array = np.array(speed_list)
    theta_array = np.array(theta_list)

# 高度ごとの統計量を求める
    # 統計データ数
    ndata = len(wind_u_array)
    sigma4 = np.zeros([nalt, 2, 2])
    # 高度ごとの風速ベクトル平均値
    wind_u_ave = np.zeros(nalt)
    wind_v_ave = np.zeros(nalt)
    sigma_xx = np.zeros(nalt)
    sigma_xy = np.zeros(nalt)
    sigma_yy = np.zeros(nalt)
    # 風速，風向平均値
    # 合成風なので注意
    speed_vec_ave = np.zeros(nalt)
    theta_vec_ave = np.zeros(nalt)
    p_log = []
    means_log = []
    w_log = []
    h_log = []
    theta_log = []
    print(nalt)
    for i in range(nalt):
        print(i)
        wind_u = np.zeros(ndata)
        wind_v = np.zeros(ndata)
        wind_u = wind_u_array[:,i]
        wind_v = wind_v_array[:,i]
        #print(np.cov(wind_u[i], wind_v[i]).shape)
        #sigma4[i,:,:]   = np.cov(wind_u[i], wind_v[i])
        wind_u_ave[i] = sum(wind_u) / ndata
        wind_v_ave[i] = sum(wind_v) / ndata
        cov_tmp = np.cov(wind_u, wind_v)
        l, v = np.linalg.eig(cov_tmp)

        el = ConfidenceEllipse((np.array([wind_u, wind_v])).transpose(), 0.95)
        p = el.get_point()
        means, w, h, theta = el.get_params()
        p_log.append(p)
        means_log.append(means)
        w_log.append(w)
        h_log.append(h)
        theta_log.append(theta)

        fig, ax = plt.subplots()
        ax.set_title('alt='+str(alt_sta+float(i)*alt_del)+' m')
        ax.scatter(wind_u_ave[i], wind_v_ave[i], color='b', marker='*')
        ax.plot(p[:,0], p[:,1], color='r', marker='o')
        ax.add_artist(
            Ellipse(xy=means,
            width=w, height=h,
            angle=theta, color='r', alpha=0.5)
        )
        ax.scatter(wind_u, wind_v, color='black')
        ax.set_aspect('equal')
        #plt.show()
        #print(i)

    #speed_vec_ave = np.sqrt(wind_u_ave**2 + wind_v_ave**2)
    #theta_vec_ave = np.arctan2(wind_u_ave, wind_v_ave) * 180.0 / np.pi
    #plt.plot(wind_u_ave, alt_array)
    #plt.plot(wind_v_ave, alt_array)
    #plt.plot(speed_vec_ave, alt_array)
    #plt.show()

    p_log = np.array(p_log)
    means_log = np.array(means_log)
    w_log = np.array(w_log)
    h_log = np.array(h_log)
    theta_log = np.array(theta_log)
    # 3D graph
    plt.close('all')
    fig1 = plt.figure('3D graph')
    origin = np.array([0.0, 0.0, 0.0])
    ax = fig1.gca(projection = '3d')
    ax.set_xlabel('u [m/s]')
    ax.set_ylabel('v [m/s]')
    ax.set_zlabel('Altitude [m]')
    ax.set_title('3D graph')
    angle = np.linspace(0.0, 2.0 * np.pi, 100)
    ell_x = np.zeros([nalt, 100])
    ell_y = np.zeros([nalt, 100])
    ell_z = np.zeros([nalt, 100])
    for i in range(nalt):
        w = w_log[i]
        h = h_log[i]
        theta = np.deg2rad(theta_log[i])
        means = means_log[i,:]
        tmp_p = np.array(
            [0.5 * w * np.cos(angle), 
            0.5 * h * np.sin(angle)]).transpose()
        rot_mat = np.array([[np.cos(theta), - np.sin(theta)],
                            [np.sin(theta),   np.cos(theta)]])
        tmp2_p = np.array([means + rot_mat.dot(p) for p in tmp_p])
        ell_x[i,:] = tmp2_p[:,0]
        ell_y[i,:] = tmp2_p[:,1]
        ell_z[i,:] = alt_array[i]
    ax.plot_wireframe(
        ell_x, ell_y, ell_z,
        rcount=nalt, ccount=8,
        color='red', 
        alpha=0.6)

    #for i in range(8):
    #    ax.plot(
    #        p_log[:,i,0],
    #        p_log[:,i,1],
    #        alt_array[:]
    #    )
    # for i in range(nalt):
    #     ax.plot(p_log[i,:,0],
    #             p_log[i,:,1],
    #             alt_array[i])
    ax.plot(means_log[:,0], means_log[:,1], alt_array, color='orange')
    ax.legend()
    ax.set_zlim(bottom=0.0)
    #fig1.savefig(self.filepath +'/'+ flightType + '/Flightlog.png')
    plt.show()

    print('----- end debug')

    df_data = pd.DataFrame({
        'altitude':alt_array,
        'wind_u_ave':wind_u_ave,
        'wind_v_ave':wind_v_ave,
        'speed_vec_ave':speed_vec_ave,
        'theta_vec_ave':theta_vec_ave
    })
    file_name = os.path.join(savedir,'test.csv')
    df_data.to_csv('test.csv', encoding='utf-8')
    
    print('OK')