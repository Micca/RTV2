/*
* Copyright (C) 2016
* Computer Graphics Group, The Institute of Computer Graphics and Algorithms, TU Wien
* Written by Tobias Klein <tklein@cg.tuwien.ac.at>
* All rights reserved.
*/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QStatusBar>
#include <QVariant>

#include "ui_MainWindow.h"
#include "GLWidget.h"
#include "streamserver.h"


class MainWindow : public QMainWindow
{
	Q_OBJECT

public:

	MainWindow(QWidget *parent = 0);
	~MainWindow();

	void displayTotalGPUMemory(float size);
	void displayUsedGPUMemory(float size);
	void displayFPS(int fps);

    void openFile(const QString &filename);

	inline GLWidget *getGLWidget()
	{
		return m_glWidget;
	}

    static inline std::vector<float> getArrivalDelays()
    {
        return arrivalDelays;
    }

    static inline std::vector<float> getDepartureDelays()
    {
        return departureDelays;
    }

protected slots :

	void openFileAction();
	void closeAction();

private:

	// USER INTERFACE ELEMENTS

	Ui_MainWindow *m_Ui;
    GLWidget *m_glWidget;

	// DATA 

    static std::vector<float> arrivalDelays;
    static std::vector<float> departureDelays;

    // NOT USED
    
	enum DataType
	{
		NETCDF
	};

	struct FileType
	{
		QString filename;
		DataType type;
	} m_FileType;

	std::vector<std::vector<Atom> > m_animation;
	

};

#endif
