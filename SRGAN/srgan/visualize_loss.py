import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import os, math, sys
from PIL import Image
from tqdm import tqdm_notebook as tqdm

if not os.path.exists("plots"):
    os.mkdir("plots")
######################################################################PLOT generator loss###############################################

df_gen_train = pd.read_csv("train_generator_losses.csv")
df_gen_test = pd.read_csv("test_generator_losses.csv")
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(df_gen_train["x"]), y=list(df_gen_train["y"]), mode='lines', name='Train Generator Loss'))
fig.add_trace(go.Scatter(x=list(df_gen_test["x"]), y=list(df_gen_test["y"]), marker_symbol='star-diamond', 
                         marker_color='orange', marker_line_width=1, marker_size=9, mode='markers', name='Test Generator Loss'))
fig.update_layout(
    width=1000,
    height=500,
    title="Train vs. Test Generator Loss",
    xaxis_title="Number of training examples seen",
    yaxis_title="Adversarial + Content Loss"),


fig.write_image("plots/Adversarial + Content Loss.png")




######################################################################PLOT discriminator loss###############################################

df_disc_train = pd.read_csv("train_discriminator_losses.csv")
df_disc_test = pd.read_csv("test_discriminator_losses.csv")

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(df_disc_train["x"]), y=list(df_disc_train["y"]), mode='lines', name='Train Discriminator Loss'))
fig.add_trace(go.Scatter(x=list(df_disc_test["x"]), y=list(df_disc_test["y"]), marker_symbol='star-diamond', 
                         marker_color='orange', marker_line_width=1, marker_size=9, mode='markers', name='Test Discriminator Loss'))
fig.update_layout(
    width=1000,
    height=500,
    title="Train vs. Test Discriminator Loss",
    xaxis_title="Number of training examples seen",
    yaxis_title="Adversarial Loss"),
fig.write_image("plots/Adversarial Loss.png")