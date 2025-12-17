# Propagator info

find the propagator model at the following repository
```
https://github.com/CIMAFoundation/propagator_sim
```

# run propagator - local

1) clone the repository 
```
git clone https://github.com/GiorgioMeschi/propagator-st.git
```

2) create a conda env and install require packages
```
cd ./
conda create --prefix .venv/ python=3.13
conda activate .venv/
pip install -r requirements.txt
```
3) run streamlit and use propagator on localhost
```  
streamlit run propagator_app.py
```



