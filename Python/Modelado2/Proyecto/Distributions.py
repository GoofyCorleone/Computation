import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from plotly.offline import iplot
from scipy.stats import vonmises_fisher
from sphere.distribution import fb8, fb8_mle, kent_me


class Distributions_Data():
    
    def __init__(self, route=False , DData=[]):
        self.route = route
        self.DData = DData
        
    def readData(self,retornar=False):
        self.Data = pd.read_csv(self.route , header= 22 , encoding= 'latin' , engine='python')
        self.Data.reset_index(inplace=True)
        
        if retornar:
            return self.Data
    
    def GetStokes(self,retS = False, retDOP = False):
        """
                GetStokes
            
            Inputs:
             - retS : It is a boolean value
             - retDOP: It is a boolean value
             
            This method uses the method readData, which allows us to manipulate
            the pandas dataframe of the working .csv, defines four empty numpy
            arrays, each one for the time, S1, S2,S3 and DOP, respectively. 
            
            
            Ouputs:
             - If the user wants the Stokes parameters and the DOP, will return both.
              from the inputs also can be returned only the DOP and not the Stokes 
              parameters and viceverse otherwise none will be returned.
        """
        self.readData()
        
        t = self.Data['index'].values
        self.time = np.array([])
        S1 , S2 , S3 = self.Data['Time Stamp [s]'].values , self.Data['Stokes 1'].values , self.Data['Stokes 2'].values
        # S1 , S2 , S3 = self.Data['Stokes 1'].values , self.Data['Stokes 2'].values , self.Data['Stokes 3'].values

        self.S1 , self.S2 , self.S3 = S1, S2, S3

        if 'Phase Difference [°]' in self.Data:
            self.DOP = self.Data['Phase Difference [°]'] / 100
        elif 'Phase Difference [ï¿½]' in self.Data:
            self.DOP = self.Data['Phase Difference [ï¿½]'] /100
        elif 'Phase Difference [�]' in self.Data:
            self.DOP = self.Data['Phase Difference [�]'] / 100
        else:
            raise Exception('Datos no compatibles para la diferencia de fase')
        # self.DOP = self.Data['Phase Difference [°]'] /100
        # self.DOP = self.Data['Phase Difference [ï¿½]'] /100
        # print('Phase Difference [ï¿½]' in self.Data)
        # self.DOP = self.Data['Phase Difference [�]'] /100
        # self.DOP = self.Data['DOP [%]']
        self.DOP = self.DOP.values
            
        if retS and retDOP:
            return self.S1 , self.S2 , self.S3 , self.DOP
        elif retS and not retDOP:
            return self.S1 , self.S2 , self.S3
        elif not retS and retDOP:
            return self.DOP
        
        
    def Get_Sphere(self,color,title):
        
        phi, theta = np.linspace(0,2*np.pi,100) , np.linspace(0,np.pi,100)
        phi , theta = np.meshgrid(phi,theta)
        x , y , z = np.cos(phi)*np.sin(theta) , np.sin(phi)*np.sin(theta) , np.cos(theta)

        trace = go.Surface(
            x = x,
            y = y,
            z = z,
            showscale=False,
            opacity=0.5,
            colorscale='Peach',
            # surfacecolor=[[color] * 100 for _ in range(100)] # Here we can define a uniform surface color uwu
        )

        layout = go.Layout(
            title=title,
            scene=dict(
                xaxis_title='S1',
                yaxis_title='S2',
                zaxis_title='S3',
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False
            )
        )
        return trace , layout
    
    def get_scatter3D(self,colorscale,title):
        scatter = go.Scatter3d(
            x=self.S1,
            y=self.S2,
            z=self.S3,
            mode='markers',
            marker=dict(
                size=2,
                color=self.DOP,  
                colorscale=colorscale, 
                cmin = min(self.DOP),
                cmax = max(self.DOP),
                # cmin=0,
                # cmax=1,
                colorbar=dict(title=title)
            ),
            showlegend=False
        )
        return scatter
    
    def get_MeridianParallel(self):
        phi, theta = np.linspace(0,2*np.pi,100) , np.linspace(0,np.pi,100)
        PHI , THETA = phi , theta
        # PHI, THETA = np.meshgrid(phi,theta)
        xm = np.sin((np.pi/2)) * np.cos(PHI)
        ym = np.sin((np.pi/2)) * np.sin(PHI)
        zm = np.cos((np.pi/2)) * np.ones(np.shape(PHI))
        
        xp = np.sin(np.ones(np.shape(THETA))*np.pi/2) * np.cos((np.pi/2))
        yp = np.sin(THETA) * np.sin((np.pi/2))
        zp = np.cos(THETA) * np.sin((np.pi/2))
        
        xp2 = np.sin(THETA) * np.cos(0)
        yp2 = np.sin(THETA) * np.sin((np.pi))
        zp2 = np.cos(THETA) * np.sin((np.pi/2))
        
        meridian_1 = go.Scatter3d(x=xm, y=ym, z=zm, mode='lines', line=dict(color='black', width=1),showlegend=False)
        
        parallel_p1 = go.Scatter3d(x=xp, y=yp, z=zp, mode='lines', line=dict(color='black', width=1),showlegend=False)
        parallel_p2 = go.Scatter3d(x=-xp, y=-yp, z=zp, mode='lines', line=dict(color='black', width=1),showlegend=False)
        
        parallel2_p1 = go.Scatter3d(x=xp2, y=yp2, z=zp2, mode='lines', line=dict(color='black', width=1),showlegend=False)
        parallel2_p2 = go.Scatter3d(x=-xp2, y=-yp2, z=zp2, mode='lines', line=dict(color='black', width=1),showlegend=False)
        return meridian_1 , parallel_p1 , parallel_p2 , parallel2_p1 , parallel2_p2
    
    def PlotMD(self,SavePDF=False,color='blue',title='Poincaré Sphere',colorscale='Inferno',titleS='DOP',PDFname='Distribution'):
        
        self.GetStokes()
        sphere , layout = self.Get_Sphere(color=color,title=title)
        scatter = self.get_scatter3D(colorscale=colorscale,title=titleS)
        line_1 = go.Scatter3d(x=[-1.0, 1.0], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=3),showlegend=False)
        line_2 = go.Scatter3d(x=[0, 0], y=[-1.0, 1.0], z=[0, 0], mode='lines', line=dict(color='black', width=3),showlegend=False)
        line_3 = go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.0, 1.0], mode='lines', line=dict(color='black', width=3),showlegend=False)
        m1 , pp1 , pp2 , p2p1 , p2p2= self.get_MeridianParallel()
        
        data = [sphere, scatter , line_1 , line_2 , line_3 , m1 , pp1 , pp2 , p2p1 , p2p2]
        fig = go.Figure(data=data , layout=layout)
        fig.update_layout(
            scene=dict(
                annotations=[
                    dict(
                        showarrow=False,
                        x=0,
                        y=0,
                        z=1.2,
                        text = 'S3'
                    ),
                    dict(
                        showarrow=False,
                        x=1.2,
                        y=0,
                        z=0,
                        text = 'S1'
                    ),
                    dict(
                        showarrow=False,
                        x=0,
                        y=1.2,
                        z=0,
                        text = 'S2'
                    )
                ]
            )
        )
        iplot(fig)
        if SavePDF:
            pio.write_image(fig, PDFname+'.pdf') #It is necessary the kaleido  librery
            
    def PlotDD(self,SavePDF=False,color='blue',title='Poincaré Sphere',colorscale='Inferno',titleS='DOP',PDFname='Distribution',ColorDOP=  []):
        
        sphere , layout = self.Get_Sphere(color=color,title=title)
        DD_DOP = np.sqrt(self.DData[:,0]**2 + self.DData[:,1]**2 + self.DData[:,2]**2)
        line_1 = go.Scatter3d(x=[-1.0, 1.0], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=2),showlegend=False)
        line_2 = go.Scatter3d(x=[0, 0], y=[-1.0, 1.0], z=[0, 0], mode='lines', line=dict(color='black', width=2),showlegend=False)
        line_3 = go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.0, 1.0], mode='lines', line=dict(color='black', width=2),showlegend=False)
        
        m1 , pp1 , pp2 , p2p1 , p2p2= self.get_MeridianParallel()
        if len(ColorDOP ) > 2 :
            scatter = go.Scatter3d(
                x=self.DData[:, 0],
                y=self.DData[:, 1],
                z=self.DData[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=ColorDOP,
                    colorscale=colorscale,
                    # cmin=0,
                    # cmax=1,
                    colorbar=dict(title='DOP')
                ),
                showlegend=False
            )
        else:
            scatter = go.Scatter3d(
                x=self.DData[:, 0],
                y=self.DData[:, 1],
                z=self.DData[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=DD_DOP,
                    colorscale=colorscale,
                    # cmin=0,
                    # cmax=1,
                    colorbar=dict(title='DOP')
                ),
                showlegend=False
            )
        data = [sphere, scatter , line_1 , line_2 , line_3 , m1 , pp1 , pp2 , p2p1 , p2p2]
        fig = go.Figure(data = data , layout= layout)
        fig.update_layout(
            scene=dict(
                annotations=[
                    dict(
                        showarrow=False,
                        x=0,
                        y=0,
                        z=1.2,
                        text = 'S3'
                    ),
                    dict(
                        showarrow=False,
                        x=1.2,
                        y=0,
                        z=0,
                        text = 'S1'
                    ),
                    dict(
                        showarrow=False,
                        x=0,
                        y=1.2,
                        z=0,
                        text = 'S2'
                    )
                ]
            )
        )
        iplot(fig)
        if SavePDF:
            pio.write_image(fig, PDFname+'.pdf')

    def Depol_Curve(self, retardances , text , xaxis_title='Retardance', autorange=True):
        self.GetStokes()
        scattDOP = go.Scatter(
            x = retardances,
            y = self.DOP,
            mode = 'markers',
            marker = dict(
                color = 'MediumPurple',
                opacity = self.DOP
            )
        )
        fig = go.Figure(data=scattDOP)
        fig.update_layout(
            title={
                'text': text,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title= xaxis_title,
            yaxis_title='DOP',
            xaxis=dict(
                autorange=autorange  # Invertir el eje X
            )

        )
        fig.show()

# mu = np.array([0,1,0.5])
# mu = mu/np.linalg.norm(mu)
# kappa = 100000
# DData = vonmises_fisher(mu=mu,kappa=kappa).rvs(200)
# try2 = Distributions_Data(DData=DData)
# try2.PlotDD()

# route = 'DiodoLedLVL6ntensity.csv'
# try1 = Distributions_Data(route=route)
# try1.PlotMD(SavePDF=False)

# route = 'Polarizado.csv'
# try1 = Distributions_Data(route=route)
# try1.PlotMD(SavePDF=False)

# route = 'Depolarizado.csv'
# try1 = Distributions_Data(route=route)
# try1.PlotMD(SavePDF=False)

# route = 'repolarizado.csv'
# try1 = Distributions_Data(route=route)
# try1.PlotMD(SavePDF=False)

# route = 'Depolarizacion/LCdepoFPC8.csv'
# try1 = Distributions_Data(route=route)
# try1.PlotMD(SavePDF=False)

# route = 'DistriLED/3.csv'
# try1 = Distributions_Data(route = route)
# try1.PlotMD()
# s1 , s2 , s3 = try1.GetStokes(retS=True)
# samples = np.ones((len(s1),3))
# samples [:,0] = s1
# samples [:,1] = s2
# samples [:,2] = s3
# k_mle = kent_me(samples)
# DData = k_mle.rvs(1024)
# mu_fit, kappa_fit = vonmises_fisher.fit(samples)
# print(mu_fit,kappa_fit)
# DData = vonmises_fisher(mu=mu_fit,kappa=kappa_fit).rvs(1024)
# try2 = Distributions_Data(DData=DData)
# try2.PlotDD()