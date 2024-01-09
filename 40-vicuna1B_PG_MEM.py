"""
Download the model
-------------------------------------
https://huggingface.co/afrideva/Tiny-Vicuna-1B-GGUF
Hugging Face repo: afrideva/Tiny-Vicuna-1B-GGUF

Tiny Vicuna 1B
This model is a fine-tuned version of TinyLlama on WizardVicuna Dataset. 
It should be fully compatible with Vicuna-v1.5 series.
This model is easy to iterate on for early experiments!

ICON from https://www.hardware-corner.net/wp-content/uploads/llm/images/Vicuna-logo.jpg
PLOTLY tutorial https://plotly.com/python/text-and-annotations/
COLOR codes from https://html-color.codes/gold/chart
PROMPT TEMPLATE RESOURCE: https://www.hardware-corner.net/llm-database/Vicuna/
MAIN: https://www.hardware-corner.net/llm-database/
CONTEXT https://github.com/fabiomatricardi/cdQnA/blob/main/KS-all-info_rev1.txt

"""
import gradio as gr
from llama_cpp import Llama
import datetime
import psutil # to get the SYSTEM MONITOR CPU/RAM stats
import pandas as pd # to visualize the SYSTEM MONITOR CPU/RAM stats

################# MODEL SETTINGS also for DISPLAY ##################
initial_RAM = psutil.virtual_memory()[2]
initial_CPU = psutil.cpu_percent() 
import plotly.express as px
plot_end = 1
data = pd.DataFrame.from_dict({"x": [0], "y": [initial_RAM],"y1":[initial_CPU]}) 


######################## MAIN VARIABLES ################3###########
liked = 2
convHistory = ''
convHistory = ''
mrepo = 'afrideva/Tiny-Vicuna-1B-GGUF'
modelfile = "./tiny-vicuna-1b.q5_k_m.gguf"
modeltitle = "TinyVicuna1B-q8-GGUF"
modelparameters = '1 B'
model_is_sys = False
modelicon = '🦙'
imagefile = 'https://www.hardware-corner.net/wp-content/uploads/llm/images/Vicuna-logo.jpg'
repetitionpenalty = 1.2
contextlength=2048
stoptoken = '<s>'  #'</s>'
logfile = f'{modeltitle}_logs.txt'
print(f"loading model {modelfile}...")
stt = datetime.datetime.now()
################ LOADING THE MODELS  ###############################
# Set gpu_layers to the number of layers to offload to GPU. 
# Set to 0 if no GPU acceleration is available on your system.
####################################################################
llm = Llama(
  model_path=modelfile,  # Download the model file first
  n_ctx=contextlength,  # The max sequence length to use - note that longer sequence lengths require much more resources
  #n_threads=2,            # The number of CPU threads to use, tailor to your system and the resulting performance
)

dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")

########## FUnCTIOn TO WRITe lOGFIle ######################
def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

######## FUNCTION FOR PLOTTING CPU RAM % ################

def get_plot(period=1):
    global plot_end
    global data
    w = 300
    h = 150
    # NEW DATA FOR THE DATAFRAME
    x = plot_end
    y = psutil.virtual_memory()[2]
    y1 = psutil.cpu_percent()
    new_record = pd.DataFrame([{'x':x, 'y':y, 'y1':y1}])
    data = pd.concat([data, new_record], ignore_index=True)
    # TO HIDE ALL PLOTLY OPTION BAR
    modebars = ["autoScale2d", "autoscale", "editInChartStudio", "editinchartstudio", "hoverCompareCartesian", "hovercompare", "lasso", "lasso2d", "orbitRotation", "orbitrotation", "pan", "pan2d", "pan3d", "reset", "resetCameraDefault3d", "resetCameraLastSave3d", "resetGeo", "resetSankeyGroup", "resetScale2d", "resetViewMapbox", "resetViews", "resetcameradefault", "resetcameralastsave", "resetsankeygroup", "resetscale", "resetview", "resetviews", "select", "select2d", "sendDataToCloud", "senddatatocloud", "tableRotation", "tablerotation", "toImage", "toggleHover", "toggleSpikelines", "togglehover", "togglespikelines", "toimage", "zoom", "zoom2d", "zoom3d", "zoomIn2d", "zoomInGeo", "zoomInMapbox", "zoomOut2d", "zoomOutGeo", "zoomOutMapbox", "zoomin", "zoomout"]
    # RAM LINE CHART
    fig = px.area(data, x="x", y='y',height=h,line_shape='spline',range_y=[0,100]) #, width=300
    fig.update_traces(line_color='#6495ed', line_width=2)
    fig.update_layout(annotations=[], overwrite=True)
    fig.update_xaxes(visible=False) #, fixedrange=False
    fig.add_annotation(text=f"<b>{y} %</b>",
                xref="paper", yref="paper",
                x=0.3, y=0.12, showarrow=False,
                font=dict(
                    family="Balto, sans-serif",
                    size=30,
                    color="#ffe02e"  #
                    ),
                align="center",)
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(t=1,l=1,b=1,r=1),
        modebar_remove=modebars
    )
    # CPU LINE CHART
    fig2 = px.area(data, x="x", y='y1',line_shape='spline',height=h,range_y=[0,100])  #, width=300 #line_shape='spline'
    fig2.update_traces(line_color='#ff5757', line_width=2)
    fig2.update_layout(annotations=[], overwrite=True)
    fig2.update_xaxes(visible=False) #, fixedrange=True
    #fig.update_yaxes(visible=False, fixedrange=True)
    # strip down the rest of the plot
    fig2.add_annotation(text=f"<b>{y1} %</b>",
                  xref="paper", yref="paper",
                  x=0.3, y=0.12, showarrow=False,
                  font=dict(
                        family="Balto, sans-serif",
                        size=30,
                        color="#ad9300"  ##ad9300
                        ),
                  align="center",)    
    fig2.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        modebar_remove=modebars
    )
    plot_end += 1 
    return fig, fig2


########### PROMPT TEMPLATE SECTION####################
"""
PROMPT TEMPLATE RESOURCES
https://www.hardware-corner.net/llm-database/Vicuna/

f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {b} ASSISTANT:"

"""
############# FUNCTION FOT THE LLM GENERATION WITH LLAMA.CPP #######################
def combine(a, b, c, d,e,f):
    global convHistory
    import datetime
    temperature = c
    max_new_tokens = d
    repeat_penalty = f
    top_p = e
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {b} ASSISTANT:"
    start = datetime.datetime.now()
    generation = ""
    delta = ""
    prompt_tokens = f"Prompt Tokens: {len(llm.tokenize(bytes(prompt,encoding='utf-8')))}"
    generated_text = ""
    answer_tokens = ''
    total_tokens = ''   
    for character in llm(prompt, 
                max_tokens=max_new_tokens, 
                stop=['Q:', stoptoken], #'<|im_end|>'  '#'  '<|endoftext|>'
                temperature = temperature,
                repeat_penalty = repeat_penalty,
                top_p = top_p,   # Example stop token - not necessarily correct for this specific model! Please check before using.
                echo=False, 
                stream=True):
        generation += character["choices"][0]["text"]

        answer_tokens = f"Out Tkns: {len(llm.tokenize(bytes(generation,encoding='utf-8')))}"
        total_tokens = f"Total Tkns: {len(llm.tokenize(bytes(prompt,encoding='utf-8'))) + len(llm.tokenize(bytes(generation,encoding='utf-8')))}"
        delta = datetime.datetime.now() - start
        seconds = delta.total_seconds()
        speed = (len(llm.tokenize(bytes(prompt,encoding='utf-8'))) + len(llm.tokenize(bytes(generation,encoding='utf-8'))))/seconds
        textspeed = f"Gen.Speed: {speed} t/s"
        yield generation, delta, prompt_tokens, answer_tokens, total_tokens, textspeed
    timestamp = datetime.datetime.now()
    textspeed = f"Gen.Speed: {speed} t/s"
    logger = f"""time: {timestamp}\n Temp: {temperature} - MaxNewTokens: {max_new_tokens} - RepPenalty: {repeat_penalty}  Top_P: {top_p}  \nPROMPT: \n{prompt}\n{modeltitle}_{modelparameters}: {generation}\nGenerated in {delta}\nPromptTokens: {prompt_tokens}   Output Tokens: {answer_tokens}  Total Tokens: {total_tokens}  Speed: {speed}\n---"""
    writehistory(logger)
    convHistory = convHistory + prompt + "\n" + generation + "\n"
    print(convHistory)
    return generation, delta, prompt_tokens, answer_tokens, total_tokens, textspeed   


# MAIN GRADIO INTERFACE
with gr.Blocks(theme='Medguy/base2') as demo:   #theme=gr.themes.Glass()  #theme='remilia/Ghostly'
    #TITLE SECTION
    with gr.Row(variant='compact'):
            with gr.Column(scale=3):            
                gr.Image(value=imagefile, 
                        show_label = False, width = 160,
                        show_download_button = False, container = False,)    #height = 160          
            with gr.Column(scale=10):
                gr.HTML("<center>"
                + "<h3>Prompt Engineering Playground!</h3>"
                + f"<h1>{modelicon} {modeltitle} - {modelparameters} parameters - {contextlength} context window</h1></center>")  
                with gr.Row():
                        with gr.Column(min_width=80):
                            gentime = gr.Textbox(value="", placeholder="Generation Time:", min_width=50, show_label=False)                          
                        with gr.Column(min_width=80):
                            prompttokens = gr.Textbox(value="", placeholder="Prompt Tkn:", min_width=50, show_label=False)
                        with gr.Column(min_width=80):
                            outputokens = gr.Textbox(value="", placeholder="Output Tkn:", min_width=50, show_label=False)            
                        with gr.Column(min_width=80):
                            totaltokens = gr.Textbox(value="", placeholder="Total Tokens:", min_width=50, show_label=False)   
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            - **Prompt Template**: Q:/A:
            - **Context Lenght**: {contextlength} tokens
            - **LLM Engine**: llama.cpp
            - **Model**: {modelicon} {modeltitle}
            - **Log File**: {logfile}
            """)
        with gr.Column(scale=2):
            plot = gr.Plot(label="RAM usage")
        with gr.Column(scale=2):
            plot2 = gr.Plot(label="CPU usage")


    # INTERACTIVE INFOGRAPHIC SECTION
    

    # PLAYGROUND INTERFACE SECTION
    with gr.Row():
        with gr.Column(scale=1):
            #gr.Markdown(
            #f"""### Tunning Parameters""")
            temp = gr.Slider(label="Temperature",minimum=0.0, maximum=1.0, step=0.01, value=0.1)
            top_p = gr.Slider(label="Top_P",minimum=0.0, maximum=1.0, step=0.01, value=0.8, visible=False)
            repPen = gr.Slider(label="Repetition Penalty",minimum=0.0, maximum=4.0, step=0.01, value=1.2)
            max_len = gr.Slider(label="Maximum output lenght", minimum=10,maximum=(contextlength-150),step=2, value=512)          

            txt_Messagestat = gr.Textbox(value="", placeholder="SYS STATUS:", lines = 1, interactive=False, show_label=False)              
            txt_likedStatus = gr.Textbox(value="", placeholder="Liked status: none", lines = 1, interactive=False, show_label=False)
            txt_speed = gr.Textbox(value="", placeholder="Gen.Speed: none", lines = 1, interactive=False, show_label=False) 
            clear_btn = gr.Button(value=f"🗑️ Clear Input", variant='primary')
            #CPU_usage = gr.Textbox(value="", placeholder="RAM:", lines = 1, interactive=False, show_label=False)
            #plot = gr.Plot(show_label=False)
            #plot2 = gr.Plot(show_label=False)

        with gr.Column(scale=4):
            txt = gr.Textbox(label="System Prompt", lines=1, interactive = model_is_sys, value = 'You are an advanced and helpful AI assistant.', visible=model_is_sys)
            txt_2 = gr.Textbox(label="User Prompt", lines=5, show_copy_button=True)
            with gr.Row():
                btn = gr.Button(value=f"{modelicon} Generate", variant='primary', scale=3)
                btnlike = gr.Button(value=f"👍 GOOD", variant='secondary', scale=1)
                btndislike = gr.Button(value=f"🤮 BAD", variant='secondary', scale=1)
                submitnotes = gr.Button(value=f"💾 SAVE NOTES", variant='secondary', scale=2)
            txt_3 = gr.Textbox(value="", label="Output", lines = 8, show_copy_button=True)
            txt_notes = gr.Textbox(value="", label="Generation Notes", lines = 2, show_copy_button=True)
                
            def likeGen():
                #set like/dislike and clear the previous Notes
                global liked
                liked = f"👍 GOOD"
                resetnotes = ""
                return liked
            def dislikeGen():
                #set like/dislike and clear the previous Notes
                global liked
                liked = f"🤮 BAD"
                resetnotes = ""
                return liked
            def savenotes(vote,text):
                logging = f"### NOTES AND COMMENTS TO GENERATION\nGeneration Quality: {vote}\nGeneration notes: {text}\n---\n\n"
                writehistory(logging)
                message = "Notes Successfully saved"
                print(logging)
                print(message)
                return message
            def clearInput(): #Clear the Input TextArea
                message = ""
                resetnotes = ""
                reset_output = ""
                return message, resetnotes, reset_output

            btn.click(combine, inputs=[txt, txt_2,temp,max_len,top_p,repPen], outputs=[txt_3,gentime,prompttokens,outputokens,totaltokens,txt_speed])
            btnlike.click(likeGen, inputs=[], outputs=[txt_likedStatus])
            btndislike.click(dislikeGen, inputs=[], outputs=[txt_likedStatus])
            submitnotes.click(savenotes, inputs=[txt_likedStatus,txt_notes], outputs=[txt_Messagestat])
            clear_btn.click(clearInput, inputs=[], outputs=[txt_2,txt_notes,txt_3])
            dep = demo.load(get_plot, None, [plot,plot2], every=2)


if __name__ == "__main__":
    demo.launch(inbrowser=True)

#psutil.cpu_percent()