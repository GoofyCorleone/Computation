import numpy as np
import pandas as pd
import plotly.graph_objects as go

def DataReadAndBar(routes , noise = 0 , static = False):
    T = 0.9254395188272199 # Transmitancia
    if static:
        Data_00 = pd.read_csv(routes[0], header=19, delimiter='\t')
        Data_450 = pd.read_csv(routes[1], header=19, delimiter='\t')
        Data_900 = pd.read_csv(routes[2], header=19, delimiter='\t')
        Data_4590 = pd.read_csv(routes[3], header=19, delimiter='\t')

        N_00 = Data_00['Counts per Bin '].values
        N_450 = Data_450['Counts per Bin '].values
        N_900 = Data_900['Counts per Bin '].values
        N_4590 = Data_4590['Counts per Bin '].values
        # N_450 , N_900 = N_900 , N_450

        if bool(noise):
            n,m = len(Data_00['Bin Number '].values), len(noise)
            noiseM = np.zeros((n,m))
            for j in range(len(noise)):
                DataFrame = pd.read_csv(noise[j], header=19, delimiter='\t')
                noiseM[:,j] = DataFrame['Counts per Bin '].values
            noiseArr = np.linspace(1,100,n)
            print(len(noiseArr))
            for i in range(len(noiseArr)):
                noiseArr[i] = round(noiseM[i,:].mean())

            N_00 = N_00 - noiseArr
            N_450 = N_450 - noiseArr
            N_900 = N_900 - noiseArr
            N_4590 = N_4590 - noiseArr




            # N_00 = N_00 - round(N_00.mean())
            # N_450 = N_450 - round(N_00.mean())
            # N_900 = N_900 - round(N_00.mean())
            # N_4590 = N_4590 - round(N_00.mean())

            S0 = N_00 + N_900
            S1 = N_00 - N_900
            S2 = 2*N_450 - N_00 - N_900
            S3 = 2*N_4590 - N_00 - N_900

            S0M = N_00.mean() + N_900.mean()
            S1M = N_00.mean() - N_900.mean()
            S2M = 2 * N_450.mean() - N_00.mean() - N_900.mean()
            S3M = 2 * N_4590.mean() - N_00.mean() - N_900.mean()

            print('Stokes mean: ', 'S1 = ',S1M/S0M , ' S2= ',S2M/S0M , ' S3 = ', S3M/S0M)
            print('Mean DOP: ', np.sqrt((S1M/S0M)**2 + (S2M/S0M)**2 + (S2M/S0M)**2))

            s1 , s2 , s3 = S1/S0 , S2/S0 , S3/S0
            s1 , s2 , s3 = s1 * T**2 , s2 * T**2 , s3 * T**2

            #print(s1.mean(),s2.mean(),s3.mean())

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=Data_00['Bin Number '],
                    y=Data_00['Counts per Bin '],
                    name='N(0,0)',
                    marker_color='blue'
                )
            )

            fig.add_trace(
                go.Bar(
                    x=Data_450['Bin Number '],
                    y=Data_450['Counts per Bin '],
                    name='N(45,0)',
                    marker_color='green'
                )
            )

            fig.add_trace(
                go.Bar(
                    x=Data_900['Bin Number '],
                    y=Data_900['Counts per Bin '],
                    name='N(90,0)',
                    marker_color='Red'
                )
            )

            fig.add_trace(
                go.Bar(
                    x=Data_4590['Bin Number '],
                    y=Data_4590['Counts per Bin '],
                    name='N(45,90)',
                    marker_color='violet'
                )
            )
            fig.update_layout(
                title="Histograma de Counts per Bin",
                xaxis_title="Bin Number",
                yaxis_title="Counts per Bin",
                barmode='group'
            )

            fig2 = go.Figure()

            fig2.add_trace(
                go.Bar(
                    x=Data_450['Bin Number '],
                    y=s1,
                    name='s1',
                    marker_color='blue'
                )
            )

            fig2.add_trace(
                go.Bar(
                    x=Data_450['Bin Number '],
                    y=s2,
                    name='s2',
                    marker_color='red'
                )
            )

            fig2.add_trace(
                go.Bar(
                    x=Data_450['Bin Number '],
                    y=s3,
                    name='s3',
                    marker_color='green'
                )
            )
            fig.update_layout(
                title='Histogram for stokes parameters ',
                xaxis_title="Bin Number",
                yaxis_title="Stokes value per Bin",
                barmode='group'
            )


            DOP = np.sqrt(s1**2 + s2**2 + s3**2)
            fig_DOP = go.Figure()

            fig_DOP.add_trace(
                go.Bar(
                    x = Data_00['Bin Number '],
                    y = DOP,
                    marker_color = 'red'
                )
            )
            fig_DOP.update_layout(
                title='Histogram for DOP  ',
                xaxis_title="Bin Number",
                yaxis_title="Degree of polarization per Bin"
            )

        fig.show()
        fig2.show()
        fig_DOP.show()

    else:
        Bin_Number = pd.read_csv(routes[0], header=19, delimiter='\t')['Bin Number '].values
        N = 0
        theta = 0
        A , B , C , D = 0 , 0 , 0 , 0

        if bool(noise):
            n,m = len(Bin_Number), len(noise)
            noiseM = np.zeros((n,m))
            for j in range(len(noise)):
                DataFrame = pd.read_csv(noise[j], header=19, delimiter='\t')
                noiseM[:,j] = DataFrame['Counts per Bin '].values
            noiseArr = np.linspace(1,100,n)
            print(len(noiseArr))
            for i in range(len(noiseArr)):
                noiseArr[i] = round(noiseM[i,:].mean())

        for route in routes:
            Pn = pd.read_csv(route, header=19, delimiter='\t')['Counts per Bin '].values
            Pn = Pn - noiseArr
            # Pn = Pn * T**2 #Tener en cuenta la transmitancia
            A += Pn
            B += Pn * np.sin(2 * theta * np.pi / 180)
            C += Pn * np.cos(4 * theta * np.pi / 180)
            D += Pn * np.sin(4 * theta * np.pi / 180)
            theta += 10
            N += 1

        print(N)
        A = A * 2 / N
        B = B * 4 / N
        C = 4 * C / N
        D = 4 * D / N

        S0 = A-C
        S1 = 2*C
        S2 = 2*D
        S3 = B

        s1 , s2 , s3 = S1/S0 , S2/S0 , S3/S0
        # s1 , s2 , s3 = s1 * T**2 , s2 * T**2 , s3 * T**2

        print('Stokes mean: ' , s1.mean() , s2.mean(), s3.mean() )
        print('Dop mean: ', np.sqrt(s1.mean()**2 + s2.mean()**2 + s3.mean()**2))

        fig2 = go.Figure()

        fig2.add_trace(
            go.Bar(
                x=Bin_Number,
                y=s1,
                name='s1',
                marker_color='blue'
            )
        )

        fig2.add_trace(
            go.Bar(
                x=Bin_Number,
                y=s2,
                name='s2',
                marker_color='red'
            )
        )

        fig2.add_trace(
            go.Bar(
                x=Bin_Number,
                y=s3,
                name='s3',
                marker_color='green'
            )
        )
        fig2.update_layout(
            title='Histogram for stokes parameters ',
            xaxis_title="Bin Number",
            yaxis_title="Stokes value per Bin",
            barmode='group'
        )

        DOP = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
        fig_DOP = go.Figure()

        fig_DOP.add_trace(
            go.Bar(
                x=Bin_Number,
                y=DOP,
                marker_color='red'
            )
        )
        fig_DOP.update_layout(
            title='Histogram for DOP  ',
            xaxis_title="Bin Number",
            yaxis_title="Degree of polarization per Bin"
        )

        fig2.show()
        fig_DOP.show()


# routes = ['N_00.txt','N_450.txt','N_900.txt','N_4590.txt']
# DataReadAndBar(routes=routes , static = True)
#
# routes = ['N00.txt','N45-0.txt','N90-0.txt','N45-90.txt']
# DataReadAndBar(routes=routes, static = True)
#
# routes = ['N_0_0.txt','N_45_0.txt','N_90_0.txt','N_45_90.txt']
# noise = ['RuidoAmbiental_{}.txt'.format(i) for i in range(1,6)]
# DataReadAndBar(routes=routes , noise = noise , static = True)
#
routes = ['LCC/N_0_0.txt','LCC/N_45_0.txt','LCC/N_90_0.txt','LCC/N_45_90.txt']
noise = ['LCC/RuidoAmbiental_{}.txt'.format(i) for i in range(1,6)]
DataReadAndBar(routes=routes , noise = noise , static = True)
#
# routes = ['Dyn1/N_{}_0.txt'.format(i) for i in np.arange(0,370,10)]
# noise = ['Dyn1/RuidoAmbiental_{}.txt'.format(i) for i in range(1,6)]
# DataReadAndBar(routes=routes , noise = noise , static = False)

# routes = ['Dyn2/N_{}_0.txt'.format(i) for i in np.arange(0,370,10)]
# noise = ['Dyn2/RuidoAmbiental_{}.txt'.format(i) for i in range(1,6)]
# DataReadAndBar(routes=routes , noise = noise , static = False)

# routes = ['Dyn3/N_{}_0.txt'.format(i) for i in np.arange(0,370,10)]
# noise = ['Dyn3/RuidoAmbiental_{}.txt'.format(i) for i in range(1,6)]
# DataReadAndBar(routes=routes , noise = noise , static = False)

# routes = ['Dyn4/N_{}_0.txt'.format(i) for i in np.arange(0,370,10)]
# noise = ['Dyn4/RuidoAmbiental_{}.txt'.format(i) for i in range(1,6)]
# DataReadAndBar(routes=routes , noise = noise , static = False)
# N_get = pd.read_csv('Dyn4/abs_con.txt', header = 19 , delimiter = '\t')
# N_sent = pd.read_csv('Dyn4/abs_sin.txt', header = 19 , delimiter = '\t')
# N_get = N_get['Counts per Bin '].values
# N_sent = N_sent['Counts per Bin '].values
# T = N_get / N_sent
# print('The transmitance is: ' , T.mean())