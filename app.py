import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow import keras

pipe = pickle.load(open('model.pkl','rb'))
player=['Hashmatullah Shahidi', 'Rashid Khan', 'Abdul Rahman','Fazalhaq Farooqi', 'Mujeeb Ur Rahman', 'Naveen-ul-Haq',
'Noor Ahmad', 'Ibrahim Zadran', 'Ikram Alikhil','Najibullah Zadran', 'Rahmanullah Gurbaz', 'Riaz Hassan',
'Azmatullah Omarzai', 'Mohammad Nabi', 'Rahmat Shah', 'AT Carey',
'GJ Maxwell', 'MP Stoinis', 'PJ Cummins', 'JR Hazlewood',
       'MA Starc', 'A Zampa', 'TM Head', 'JP Inglis', 'M Labuschagne',
       'SPD Smith', 'DA Warner', 'SA Abbott', 'C Green', 'MR Marsh',
       'Najmul Hossain Shanto', 'Hasan Mahmud', 'Mustafizur Rahman',
       'Nasum Ahmed', 'Shoriful Islam', 'Tanzim Hasan Sakib',
       'Taskin Ahmed', 'Anamul Haque', 'Litton Das', 'Mushfiqur Rahim',
       'Tanzid Hasan', 'Towhid Hridoy', 'Mahedi Hasan', 'Mahmudullah',
       'Mehidy Hasan Miraz', 'BFW de Leede', 'CN Ackermann', 'JC Buttler',
       'DJ Willey', 'CR Woakes', 'AAP Atkinson', 'AU Rashid', 'MA Wood',
       'SM Curran', 'JM Bairstow', 'HC Brook', 'DJ Malan', 'JE Root',
       'MM Ali', 'BA Carse', 'LS Livingstone', 'BA Stokes',
       'Ishan Kishan', 'JJ Bumrah', 'KL Rahul', 'Kuldeep Yadav',
       'LV van Beek', "MP O'Dowd", 'Mohammed Shami', 'Mohammed Siraj',
       'PA van Meekeren', 'A Dutt', 'RE van der Merwe', 'NRJ Croes',
       'Saqib Zulfiqar', 'Shariz Ahmad', 'Vikramjit Singh',
       'KS Williamson', 'MJ Santner', 'TA Boult', 'LH Ferguson',
       'KA Jamieson', 'IS Sodhi', 'TG Southee', 'TWM Latham', 'DP Conway',
       'GD Phillips', 'WA Young', 'MS Chapman', 'DJ Mitchell',
       'JDS Neesham', 'R Ravindra', 'Babar Azam', 'Mohammad Nawaz (3)',
       'Shadab Khan', 'Haris Rauf', 'Hasan Ali', 'Shaheen Shah Afridi',
       'Usama Mir', 'Fakhar Zaman', 'Imam-ul-Haq', 'Saud Shakeel',
       'Abdullah Shafique', 'Iftikhar Ahmed', 'Mohammad Rizwan',
       'Agha Salman', 'Mohammad Wasim (1)', 'M Prasidh Krishna',
       'RA Jadeja', 'RG Sharma', 'R Ashwin', 'T Bavuma', 'G Coetzee',
       'L Ngidi', 'T Shamsi', 'K Rabada', 'KA Maharaj', 'RR Hendricks',
       'AK Markram', 'HE van der Dussen', 'DA Miller', 'H Klaasen',
       'Q de Kock', 'M Jansen', 'AL Phehlukwayo', 'SA Edwards',
       'SN Thakur', 'SS Iyer', 'Shubman Gill', 'BKG Mendis', 'AD Mathews',
       'PVD Chameera', 'D Madushanka', 'CAK Rajitha', 'M Theekshana',
       'DN Wellalage', 'FDM Karunaratne', 'P Nissanka', 'MDKJ Perera',
       'S Samarawickrama', 'KIC Asalanka', 'DM de Silva', 'MADI Hemantha',
       'C Karunaratne', 'SA Yadav', 'SA Engelbrecht', 'AT Nidamanuru',
       'V Kohli', 'W Barresi']
opposition = ['Bangladesh', 'Sri Lanka', 'Pakistan', 'India', 'Australia', 'New Zealand',
 'South Africa', 'England', 'Netherlands', 'Afghanistan']

ground = ['Mirpur', 'Abu Dhabi', 'Dubai (DSC)', 'Bristol', 'Cardiff',
       'Taunton', 'Manchester', 'Southampton', 'Leeds', 'Doha',
       'Chattogram', 'Pallekele', 'Hambantota', 'Colombo (RPS)', 'Lahore',
       'Dharamsala', 'Delhi', 'Chennai', 'Pune', 'Lucknow', 'Wankhede',
       'Ahmedabad', 'Sharjah', 'Fatullah', 'Canberra', 'Dunedin', 'Perth',
       'Napier', 'Sydney', 'Amstelveen', 'Rotterdam', 'Brisbane',
       'Chester-le-Street', 'Adelaide', 'Hobart', 'Melbourne',
       'Hyderabad', 'Nagpur', 'Ranchi', 'Mohali', 'The Oval',
       'Nottingham', "Lord's", 'Birmingham', 'Rajkot', 'Bengaluru',
       'Paarl', 'Bloemfontein', 'Potchefstroom', 'Cairns', 'Centurion',
       'Johannesburg', 'Indore', 'Jaipur', 'Harare', 'Auckland',
       'Wellington', 'Hamilton', 'Providence', 'Eden Gardens', 'Gqeberha',
       'Durban', 'Dambulla', 'Cape Town', 'Visakhapatnam', 'Basseterre',
       'Christchurch', 'Nelson', 'Colombo (SSC)', 'Dublin', 'East London',
       'Kimberley', 'Port of Spain', 'North Sound', 'Bridgetown',
       'Colombo (PSS)', 'Queenstown', 'Karachi', 'Benoni', 'Glasgow',
       'Faisalabad', 'Multan', 'Darwin', 'Mount Maunganui', 'Bulawayo',
       'Cuttack', 'Kochi', 'Raipur', 'Guwahati', 'Thiruvananthapuram',
       'Kingston', 'Gwalior', 'Vadodara', 'Kanpur', 'Rawalpindi',
       'Belfast', 'The Hague']

st.title('Cricket Match Score Predictor for Player')

Player = st.selectbox('Select Player',sorted(player))
col1, col2 = st.columns(2)
with col1:
    Opposition = st.selectbox('Select Opposition', sorted(opposition))
with col2:
    Ground = st.selectbox('Select Ground', sorted(ground))


col3,col4 = st.columns(2)

with col3:
    avg_4s = st.number_input('Average_4s',min_value=0.000,step=0.001)
with col4:
    avg_6s = st.number_input('Average_6s',min_value=0.0)

col5,col6,col7 = st.columns(3)

with col5:
    avg_bf = st.number_input('Average_ball_faced',min_value=0.000,step=0.001)
with col6:
    avg_sr = st.number_input('Average_strike_rate',min_value=0.000,step=0.001)
with col7:
    avg_mins = st.number_input('Average_minutes',min_value=0.000,step=0.001)


if st.button('Predict Score'):
    df = pd.DataFrame({f'player_{name}': [0] for name in player})
    df1 = pd.DataFrame({f'opposition_{name}': [0] for name in opposition})
    df = pd.concat([df, df1], axis=1)
    df1 = pd.DataFrame({f'ground_{name}': [0] for name in ground})
    df = pd.concat([df, df1], axis=1)
    str1 = f'player_{Player}'
    df[str1]=1
    str1 = f'opposition_{Opposition}'
    df[str1]=1
    str1 = f'ground_{Ground}'
    df[str1]=1
    df['avg_4s']=avg_4s
    df['avg_6s']=avg_6s
    df['avg_bf']=avg_bf
    df['avg_sr']=avg_sr
    df['avg_mins']=avg_mins
    result = pipe.predict(df)
    st.header("Predicted Runs " + str(int(result[0])))


