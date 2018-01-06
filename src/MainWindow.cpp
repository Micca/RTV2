/*
* Copyright (C) 2016
* Computer Graphics Group, The Institute of Computer Graphics and Algorithms, TU Wien
* Written by Tobias Klein <tklein@cg.tuwien.ac.at>
* All rights reserved.
*/

#include "MainWindow.h"

#include <fstream>
#include <string>
#include <iostream>

#include <QFileDialog>
#include <qmessagebox.h>
#include <QPainter>
#include <QXmlStreamReader>
#include <QDomDocument>


std::vector<float> MainWindow::departureDelays;
std::vector<float> MainWindow::arrivalDelays;

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
{
	m_Ui = new Ui_MainWindow();
	m_Ui->setupUi(this);
	QLayout *layout = m_Ui->controls->layout();
	layout->setAlignment(Qt::AlignTop);
	m_Ui->controls->setLayout(layout);
		

	m_glWidget = new GLWidget(this, this);
	m_Ui->glLayout->addWidget(m_glWidget);
	

	connect(m_Ui->actionOpen, SIGNAL(triggered()), this, SLOT(openFileAction()));
	connect(m_Ui->actionClose, SIGNAL(triggered()), this, SLOT(closeAction()));

	m_Ui->memSizeLCD->setPalette(Qt::darkGreen);
	m_Ui->usedMemLCD->setPalette(Qt::darkGreen);
	m_Ui->fpsLCD->setPalette(Qt::darkGreen);

}

MainWindow::~MainWindow()
{
}


void MainWindow::openFile(const QString &filename)
{
    bool success = false;
    int fileLength = 0;
    int count = 0;

    if (!filename.isEmpty()) {

        std::ifstream in((std::string) filename.toStdString(), std::ifstream::binary | std::ios::ate);

        if (in) {

            fileLength = in.tellg(); // get length of file
            in.clear();
            in.seekg(0, std::ios::beg);

            int oneArraySize = (fileLength / sizeof(int)) / 2; // byte to int32
            MainWindow::departureDelays.reserve(oneArraySize);
            MainWindow::arrivalDelays.reserve(oneArraySize);

            // TODO remove this restriction when GPU algorithm is implemented
            int upto = 500; // only take subset of data (otherwise it takes a long time). 0 means no limit.

            while (!in.eof()) {

                if (upto > 0 && count >= upto)
                    break;

                int tmp;

                in.read((char*)&tmp, sizeof(int));
                if (!in.eof()) { // reading to end of file, possibly eof bit isn't set at the loop end.
                    MainWindow::departureDelays.push_back(tmp);
                }

                in.read((char*)&tmp, sizeof(int));
                if (!in.eof()) {
                    MainWindow::arrivalDelays.push_back(tmp);
                }
                count += 1;
            }

            success = true;

        }
        in.close();
    }

    // status message
    if (success) {
        m_Ui->labelTop->setText(QString("LOADED %1 of %2 elements from file [%3]").arg(count).arg(fileLength).arg(filename));
    }
    else {
        m_Ui->labelTop->setText("ERROR loading file " + filename + "!");
    }

}

void MainWindow::openFileAction()
{
    QString filename = QFileDialog::getOpenFileName(this, "Data File", 0, 0, 0, QFileDialog::DontUseNativeDialog);
    if (!filename.isEmpty()) {
        openFile(filename);
    }
}

void MainWindow::closeAction()
{
	close();
}

void MainWindow::displayTotalGPUMemory(float size)
{
	m_Ui->memSizeLCD->display(size);
}
void MainWindow::displayUsedGPUMemory(float size)
{
	m_Ui->usedMemLCD->display(size);
}

void MainWindow::displayFPS(int fps)
{
	m_Ui->fpsLCD->display(fps);
}
