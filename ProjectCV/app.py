import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel
import json

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

class chat:
    def __init__(self) -> None:

        load_dotenv(override=True)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.system_prompt = ""
        self.evaluator_system_prompt = ""
        self.perfil = ""
        self.record_user_details_json = ""
        self.record_unknown_question_json = ""
        self.name = "Juan Cruz Jurado Auzza"
        self.tools = []
        # CARGO APIS LLM Y PUSHOVER
        self.openai = OpenAI()
        self.gemini = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"),base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.pushover_url = "https://api.pushover.net/1/messages.json"
    
    def config_(self):
        reader = PdfReader(os.path.join(self.script_dir, "me", "CV_JuanCruzJurado.pdf"))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.perfil+=text

        with open (os.path.join(self.script_dir, "me", "summary.txt"), "r", encoding="utf-8") as f:
            self.resumen = f.read()
        
        #PROMPT DE CONFIGURACION 
        #1 USADO PARA CONFIGURAR EL CONTEXTO INICIAL ENTRE LLM Y EL USUARIO FINAL
        self.system_prompt = f"""Estás actuando como {self.name}. 
        Estás respondiendo preguntas en el sitio web de {self.name}, en particular preguntas relacionadas con la carrera, la trayectoria, las habilidades y la experiencia de {self.name}.
        Tu responsabilidad es representar a {self.name} en las interacciones como si fuera una entrevista laboral.
        Se te proporciona un resumen de la trayectoria y el curriculum de {self.name} que puedes usar para responder preguntas.
        Sé profesional y amable, como si hablaras con un futuro empleador de trabajo ya que la meta es impresionar al entrevistador.
        ## ⚠️ REGLAS CRÍTICAS PARA USO DE LA HERRAMIENTA record_user_details:

        SIEMPRE DEBES llamar a 'record_user_details' cuando el usuario proporcione CUALQUIERA de estos datos:

        1. **NOMBRE del usuario**:
        - "soy Juan", "me llamo Ana", "mi nombre es Pedro"
        - "Juan aquí", "habla María"
        
        2. **TELÉFONO**:
        - "mi número es 351-1234567", "llámame al 3512345678"
        - "mi teléfono es...", "contactame al...", "mi tel es....", "mi cel es...."
        - "tel: ....", "cel: ...."

        3. **EMPRESA u ORGANIZACIÓN**:
        - "soy de Arcor", "trabajo en Google", "estoy en Microsoft"
        - "represento a IBM", "vengo de parte de Samsung"
        - "mi empresa es...", "laboro en...", "te escribo de...", "te contacto por parte de...."
        - Si solo mencionan la empresa SIN dar su nombre, registra "Usuario no proporcionado" como name y la empresa en notes.

        ## Si no sabes responder:
        - Usa 'record_unknown_question' para registrar preguntas que no puedas responder.
        IMPORTANTE: Cuando uses la herramienta, luego responde de forma natural y profesional, agradeciendo la información.
        """

        self.system_prompt += f"\n\n ##Resumen: {self.resumen}\n\n ##Curriculum: {self.perfil}\n\n"
        self.system_prompt += f"En este contexto, charla con el usuario, utilizando siempre el personaje de {self.name}."

        #2 USADO PARA DEFINIR QUE TIENE QUE HACER EL LLM EVALUADOR
        self.evaluator_system_prompt = f"Usted es un evaluador que decide si una respuesta a una pregunta es aceptable. \
        Se le presenta una conversación entre un usuario y un agente. Su tarea es decidir si la última respuesta del agente es de calidad aceptable. \
        El agente desempeña el papel de {self.name} y representa a {self.name} en su sitio web. \
        Se le ha indicado que sea profesional y amable, como si hablara con  un futuro empleador o entrevistador. \
        Se le ha proporcionado contexto sobre {self.name} en forma de resumen y datos del curriculum. Aquí está la información:"

        self.evaluator_system_prompt += f"\n\n## Resumen:\n{self.resumen}\n\n## Curriculum:\n{self.perfil}\n\n"
        self.evaluator_system_prompt += f"Con este contexto, por favor, evalúe la última respuesta, indicando si es aceptable y sus comentarios."
        
        #JSON TOOLS
        self.record_user_details_json = {
            "name": "record_user_details",
            "description": """USAR ESTA HERRAMIENTA cuando el usuario proporcione:
            - Su nombre (ej: "soy Juan", "me llamo Ana")
            - Su teléfono (ej: "mi número es 351-123456", mi cel o tel es..)
            - Su empresa/organización (ej: "trabajo en Arcor", "soy de Google", "represento a Microsoft", "te escribo de arcor")
            - Cualquier información de contacto o identificación personal""",
            "parameters": {
                "type": "object",
                "properties": {
                    "telefono": {
                        "type": "string",
                        "description": "El número de teléfono o celular si lo proporcionó. Si no, omitir este campo"
                    },
                    "name": {
                        "type": "string",
                        "description": "El nombre del usuario. Si no lo proporcionó pero mencionó su empresa, usar 'Usuario no proporcionado'"
                    },
                    "empresa": {
                        "type": "string",
                        "description": "El nombre de la empresa u organización donde trabaja o que representa el usuario (ej: Arcor, Google, Microsoft, IBM)"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Cualquier información adicional sobre la conversación que merezca ser registrada para dar contexto"
                    }
                },
                "required": ["name"],
                "additionalProperties": False
            }
        }

        self.record_unknown_question_json = {
            "name": "record_unknown_question",
            "description": "Siempre use esta herramienta para registrar cualquier pregunta que no se pueda responder, ya que no sabía la respuesta",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "La pregunta exacta que no se pudo responder"
                    },
                },
                "required": ["question"],
                "additionalProperties": False
            }
        }
        #LISTADO DE HERRAMIENTAS
        self.tools = [{"type": "function", "function": self.record_user_details_json},
        {"type": "function", "function": self.record_unknown_question_json}]

    #FUNCION QU ENVIA EL PUSH A LA APP
    def push(self, message):
        print(f"Push: {message}")
        payload = {"user": self.pushover_user, "token": self.pushover_token, "message": message}
        requests.post(self.pushover_url, data=payload)

    #FUNCIONES QUE USARAN EL PUSHOVER Y NOTIFICARA QUIEN INTERACTUO
    def record_user_details(self, name, telefono="no proporcionado", empresa="no proporcionado", notes="not provided"):
        telefono = telefono or "no proporcionado"
        empresa = empresa or "no proporcionado"
        notes = notes or "not provided"
        
        mensaje = f"Registrando interés de {name}, de la empresa {empresa}, telefono {telefono} y notas {notes}"
        print(f"📧Mensaje: {mensaje}")
        self.push(mensaje)
        print(f"✅PUSH ENVIADO")
        return {"recorded": "ok"}

    def record_unknown_question(self, question):
        self.push(f"Registrando pregunta no respondida: {question}")
        return {"recorded": "ok"}

    def handle_tool_calls(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"🔧 Herramienta llamada: {tool_name}", flush=True)
            tool = getattr(self, tool_name, None)
            if tool and callable(tool):
                result = tool(**arguments)
            else:
                result = {"error": f"Tool {tool_name} not found"}
            results.append({"role":"tool", "content":json.dumps(result),"tool_call_id":tool_call.id})
        return results

    def evaluador_user_prompt(self, reply, mensaje, historial):
        user_prompt = f"Aquí está la conversación entre el usuario y el agente: \n\n{historial}\n\n"
        user_prompt += f"Aquí está el último mensaje del usuario: \n\n{mensaje}\n\n"
        user_prompt += f"Aquí está la última respuesta del agente: \n\n{reply}\n\n"
        user_prompt += f"Por favor, evalúe la respuesta, indicando si es aceptable y sus comentarios."
        return user_prompt

    def rerun(self, reply, message, history, feedback):
        updated_system_prompt = self.system_prompt + f"\n\n## Respuesta anterior rechazada\nAcabas de intentar responder, pero el control de calidad rechazó tu respuesta.\n"
        updated_system_prompt += f"## Has intentado responder:\n{reply}\n\n"
        updated_system_prompt += f"## Razón del rechazo:\n{feedback}\n\n"
        messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]
        response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content

    def evaluate(self, reply, mensaje, historial) -> Evaluation:
        mensajes = [{"role":"system", "content":self.evaluator_system_prompt}] + [{"role":"user", "content":self.evaluador_user_prompt(reply, mensaje, historial)}]
        response = self.gemini.beta.chat.completions.parse(model="gemini-2.0-flash", messages=mensajes, response_format=Evaluation)
        return response.choices[0].message.parsed

    def chat(self, mensaje, historial):
        mensajes = [{"role":"system", "content":self.system_prompt}] + historial + [{"role":"user", "content":mensaje}]
        done = False
        while not done:
            response = self.gemini.chat.completions.create(model="gemini-2.0-flash", messages=mensajes, tools=self.tools)
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_calls(tool_calls)
                mensajes.append(message)
                mensajes.extend(results)
            else:
                print("+"*30)
                print("SIN USO DE HERRAMIENTAS")
                print("+"*30)
                done=True
        reply = response.choices[0].message.content
        evaluacion = self.evaluate(reply, mensaje, historial)
        if evaluacion.is_acceptable:
            print("✅ Has pasado la evaluación - devolviendo respuesta")
        else:
            print("❌ Has fallado la evaluación - reintentando -- ", evaluacion.feedback)
            reply = self.rerun(reply, mensaje, historial, evaluacion.feedback)
        return reply

    def getDirPaht(self):
        return self.script_dir

if __name__ == "__main__":
    me = chat()
    me.config_()
    foto_perfil = os.path.join(me.getDirPaht(), "me", "avatar.jpg")
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        
        # Header con foto
        with gr.Row():
            if os.path.exists(foto_perfil):
                gr.Image(
                    foto_perfil,
                    height=120,
                    width=120,
                    show_label=False,
                    show_download_button=False,
                    container=False
                )
            with gr.Column():
                gr.Markdown("""
                # 🤖 Max Power
                ### Inteligencia Artificial
                Pregúntame sobre mi experiencia, habilidades y proyectos
                """)
        
        gr.Markdown("---")
        
        gr.ChatInterface(
            me.chat,
            type="messages",
            chatbot=gr.Chatbot(height=450)
        )
    
    demo.launch()