// NOT NEEDED IN TASK2

/*
* Copyright (C) 2016
* Computer Graphics Group, The Institute of Computer Graphics and Algorithms, TU Wien
* Written by Tobias Klein <tklein@cg.tuwien.ac.at>
* All rights reserved.
*/

#include "GLWidget.h"

#include <qopenglwidget.h>
#include <QMouseEvent>
#include <QDir>
#include<gl/GLU.h>
#include<glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glsw.h"
#include "MainWindow.h"

const float msPerFrame = 50.0f;

#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

typedef struct {
	GLuint modelViewMatrix;
	GLuint projMatrix;
	GLuint nearPlane;
	GLuint texture_AmbOccl;
	GLuint texture_ShadowMap;
	GLuint contourEnabled;
	GLuint ambientOcclusionEnabled;
	GLuint contourConstant;
	GLuint contourWidth;
	GLuint contourDepthFactor;
	GLuint ambientFactor;
	GLuint ambientIntensity;
	GLuint diffuseFactor;
	GLuint specularFactor;
	GLuint shadowModelViewMatrix;
	GLuint shadowProjMatrix;
	GLuint lightVec;
	GLuint shadowEnabled;
} ShaderUniformsMolecules;

static ShaderUniformsMolecules UniformsMolecules;



GLWidget::GLWidget(QWidget *parent, MainWindow *mainWindow)
	: QOpenGLWidget(parent)
{
	m_MainWindow = mainWindow;
	m_fileWatcher = new QFileSystemWatcher(this);
	connect(m_fileWatcher, SIGNAL(fileChanged(const QString &)), this, SLOT(fileChanged(const QString &)));


	// watch all shader of the shader folder 
	// every time a shader changes it will be recompiled on the fly
	QDir shaderDir(QCoreApplication::applicationDirPath() + "/../../src/shader/");	
	QFileInfoList files = shaderDir.entryInfoList();
	qDebug() << "List of shaders:";
	foreach(QFileInfo file, files) {
		if (file.isFile()) {
			qDebug() << file.fileName();
			m_fileWatcher->addPath(file.absoluteFilePath());
		}
	}
	initglsw();

	renderMode = RenderMode::NONE;
	isImposerRendering = true;

	m_mrAtoms = 0;

	ambientFactor = 0.05f;
	diffuseFactor = 0.5f;
	specularFactor = 0.3f;
	m_currentFrame = 0;
}


GLWidget::~GLWidget()
{
	glswShutdown();
}

void GLWidget::initglsw()
{
	glswInit();
	QString str = QCoreApplication::applicationDirPath() + "/../../src/shader/";
	QByteArray ba = str.toLatin1();
	const char *shader_path = ba.data();
	glswSetPath(shader_path, ".glsl");
	glswAddDirectiveToken("", "#version 330");
}

void GLWidget::cleanup()
{
	// makes the widget's rendering context the current OpenGL rendering context
	makeCurrent();
	//m_vao.destroy
	m_program_molecules = 0;
	doneCurrent();
}

void GLWidget::initializeGL()
{
	connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &GLWidget::cleanup);

	QWidget::setFocusPolicy(Qt::FocusPolicy::ClickFocus);

	initializeOpenGLFunctions();
	glClearColor(0.862f, 0.929f, 0.949f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	if (!m_vao_molecules.create()) {
		qDebug() << "error creating vao";
	}
	
	m_program_molecules = new QOpenGLShaderProgram();
	m_vertexShader = new QOpenGLShader(QOpenGLShader::Vertex);
	m_geomShader = new QOpenGLShader(QOpenGLShader::Geometry);
	m_fragmentShader = new QOpenGLShader(QOpenGLShader::Fragment);


	GLint total_mem_kb = 0;
	/*glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX,
		&total_mem_kb);*/

	GLint cur_avail_mem_kb = 0;
	/*glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX,
		&cur_avail_mem_kb);*/

	float cur_avail_mem_mb = float(cur_avail_mem_kb) / 1024.0f;
	float total_mem_mb = float(total_mem_kb) / 1024.0f;

	m_MainWindow->displayTotalGPUMemory(total_mem_mb);
	m_MainWindow->displayUsedGPUMemory(0);

	connect(&mPaintTimer, SIGNAL(timeout()), this, SLOT(update()));
	mPaintTimer.start(16); // about 60FPS
	m_fpsTimer.start();
}




void GLWidget::moleculeRenderMode(std::vector<std::vector<Atom> > *animation)
{
	// makes the widget's rendering context the current OpenGL rendering context
	makeCurrent();

	m_animation = animation;

	renderMode = RenderMode::NETCDF;


	//todo: uncomment after shader is correctly loaded
//	m_program_molecules->bind();

	loadMoleculeShader();

	QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao_molecules);
	QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();

	// TODO: bind positions etc.


	
	//todo: uncomment after shader is correctly loaded
//	m_program_molecules->release();

	allocateGPUBuffer(0);
}

void GLWidget::allocateGPUBuffer(int frameNr)
{
	// makes the widget's rendering context the current OpenGL rendering context
	makeCurrent();


	//load atoms
	m_mrAtoms = (*m_animation)[frameNr].size();

	m_pos.clear();
	m_radii.clear();
	m_colors.clear();

	for (size_t i = 0; i < m_mrAtoms; i++)
	{
		Atom atom = (*m_animation)[frameNr][i];
		m_pos.push_back(atom.position);
		m_radii.push_back(atom.radius);
		m_colors.push_back(atom.color);
	}


	QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao_molecules);
	QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();

	// TODO: allocate data (positions, radii, colors)
	

	GLint total_mem_kb = 0;
	/*glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX,
		&total_mem_kb);*/

	GLint cur_avail_mem_kb = 0;
	/*glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX,
		&cur_avail_mem_kb);*/

	float cur_avail_mem_mb = float(cur_avail_mem_kb) / 1024.0f;
	float total_mem_mb = float(total_mem_kb) / 1024.0f;

	m_MainWindow->displayUsedGPUMemory(total_mem_mb - cur_avail_mem_mb);
}

bool GLWidget::loadMoleculeShader()
{
	//TODO: This has been commented out as the example shader contains just some basic information
	/*

	bool success = false;

	const char *vs = glswGetShader("molecules.Vertex");
	success = m_vertexShader->compileSourceCode(vs);

	const char *gs = glswGetShader("molecules.Geometry");
	success = m_geomShader->compileSourceCode(gs);

	const char *fs = glswGetShader("molecules.Fragment");
	success = m_fragmentShader->compileSourceCode(fs);

	m_program_molecules->addShader(m_vertexShader);
	m_program_molecules->addShader(m_geomShader);
	m_program_molecules->addShader(m_fragmentShader);
	m_program_molecules->link();

	*/
	
	

	return true;
}


void GLWidget::paintGL()
{
	calculateFPS();
	switch (renderMode) {
	case(RenderMode::NONE):
		break; // do nothing
	case(RenderMode::PDB):
		// TODO:
		break;
	case(RenderMode::NETCDF):
		drawMolecules();
		break;
	default:
		break;

	}
}


void GLWidget::drawMolecules() 
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// animate frames
	if (m_isPlaying) {
		qint64 elapsed = m_AnimationTimer.elapsed() - m_lastTime;
		
		elapsed -= msPerFrame;
		while (elapsed > 0) {
			m_currentFrame++;
			m_lastTime = m_AnimationTimer.elapsed();
			elapsed -= msPerFrame;
			
		}
		if (m_currentFrame >= (*m_animation).size()) {
			m_currentFrame = (*m_animation).size() - 1;
			m_isPlaying = false;
		}
		

		allocateGPUBuffer(m_currentFrame);
	}


	if (isImposerRendering) {
		//todo:implement
		// bind vertex array object and program


		// set uniforms
		

		// draw call

	}
	else { 
		
		//simplistic implementation. 
		
		
		size_t m_mrAtoms = (*m_animation)[m_currentFrame].size();


		ambientFactor = 0.05f;
		diffuseFactor = 0.5f;
		specularFactor = 0.3f;


		GLfloat light_ambient[] = { ambientFactor,ambientFactor,ambientFactor, 1.0f };
		GLfloat light_diffuse[] = { diffuseFactor, diffuseFactor, diffuseFactor, 1.0f };
		GLfloat light_specular[] = { specularFactor, specularFactor, specularFactor, 1.0f };

		GLfloat light_position[] = { 0.0f, 0.0f, -100.0f, 1.0f };


		//the folliwng is not that important.

		glLoadIdentity();
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_LIGHTING);
		
		glDepthFunc(GL_LEQUAL);
		glClearDepth(1.0f);

		glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
		glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
		glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

		glEnable(GL_LIGHT0);

		glEnable(GL_COLOR_MATERIAL);
		glColorMaterial(GL_FRONT, GL_AMBIENT);
		glColorMaterial(GL_FRONT, GL_DIFFUSE);
		glColorMaterial(GL_FRONT, GL_SPECULAR);
		glMaterialf(GL_FRONT, GL_SHININESS, 128.0f);
		

		
		//transformations etc.


		glLoadIdentity();
		GLenum er = glGetError();
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(glm::value_ptr(m_camera.getProjectionMatrix()));
		glMatrixMode(GL_MODELVIEW);
		glMultMatrixf(glm::value_ptr(m_camera.getViewMatrix()));

		glLightfv(GL_LIGHT0, GL_POSITION, light_position);


		er = glGetError();

		for (size_t i = 0; i < m_mrAtoms; i++) { //
			Atom atom = (*m_animation)[m_currentFrame][i];
			glPushMatrix();
			GLUquadric *quad;
			quad = gluNewQuadric();
			//set color and position
			glColor4f(atom.color.r, atom.color.g, atom.color.b, 1);
			glTranslatef(atom.position.x, atom.position.y, atom.position.z);
			er = glGetError();
			gluSphere(quad, atom.radius, 40, 40);
			er = glGetError();
			glPopMatrix();
			gluDeleteQuadric(quad);
		}
	}



}

void GLWidget::calculateFPS()
{
	m_frameCount++;

	qint64 currentTime = m_fpsTimer.elapsed();

	//  Calculate time passed
	qint64 timeInterval = currentTime - m_previousTimeFPS;

	if (timeInterval > ((qint64)1000))
	{
		// calculate the number of frames per second
		m_fps = m_frameCount / (timeInterval / 1000.0f);

		// set time
		m_previousTimeFPS = currentTime;

		// reset frame count
		m_frameCount = 0;
	}

	m_MainWindow->displayFPS(m_fps);
}



void GLWidget::resizeGL(int w, int h)
{
	m_camera.setAspect(float(w) / h);

}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
	m_lastPos = event->pos();
}

void GLWidget::wheelEvent(QWheelEvent *event)
{
	m_camera.zoom(event->delta() / 30);
	update();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
	int dx = event->x() - m_lastPos.x();
	int dy = event->y() - m_lastPos.y();

	if (event->buttons() & Qt::LeftButton) {
		m_camera.rotateAzimuth(dx / 100.0f);
		m_camera.rotatePolar(dy / 100.0f);
	}

	if (event->buttons() & Qt::RightButton) {
		m_camera.rotateAzimuth(dx / 100.0f);
		m_camera.rotatePolar(dy / 100.0f);
	}
	m_lastPos = event->pos();
	update();
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
	switch (event->key())
	{
	case Qt::Key_Space:
	{
		break;
	}
	default:
	{
		event->ignore();
		break;
	}
	}
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{

}

void GLWidget::fileChanged(const QString &path)
{
	// reboot glsw, otherwise it will use the old cached shader
	glswShutdown();
	initglsw();

	loadMoleculeShader();
	update();
}

void GLWidget::playAnimation()
{
	m_AnimationTimer.start();
	m_lastTime = 0;
	m_isPlaying = true;
}

void GLWidget::pauseAnimation()
{
	m_isPlaying = false;
}

void GLWidget::setAnimationFrame(int frameNr)
{
	m_currentFrame = frameNr;
	allocateGPUBuffer(frameNr);
}
