/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpen;
    QAction *actionClose;
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    QWidget *controls;
    QGridLayout *gridLayout_2;
    QProgressBar *progressBar;
    QLabel *labelTop;
    QGridLayout *glLayout;
    QFrame *line;
    QWidget *widget_2;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox_5;
    QLabel *label_9;
    QLCDNumber *usedMemLCD;
    QLabel *label_10;
    QLabel *label_11;
    QLCDNumber *memSizeLCD;
    QLCDNumber *fpsLCD;
    QLabel *label_12;
    QMenuBar *menubar;
    QMenu *menuFile;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1112, 748);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainWindow->sizePolicy().hasHeightForWidth());
        MainWindow->setSizePolicy(sizePolicy);
        actionOpen = new QAction(MainWindow);
        actionOpen->setObjectName(QStringLiteral("actionOpen"));
        actionClose = new QAction(MainWindow);
        actionClose->setObjectName(QStringLiteral("actionClose"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        controls = new QWidget(centralwidget);
        controls->setObjectName(QStringLiteral("controls"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(controls->sizePolicy().hasHeightForWidth());
        controls->setSizePolicy(sizePolicy1);
        gridLayout_2 = new QGridLayout(controls);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        progressBar = new QProgressBar(controls);
        progressBar->setObjectName(QStringLiteral("progressBar"));
        progressBar->setEnabled(false);
        QSizePolicy sizePolicy2(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(progressBar->sizePolicy().hasHeightForWidth());
        progressBar->setSizePolicy(sizePolicy2);
        progressBar->setMaximum(100);
        progressBar->setValue(0);
        progressBar->setTextVisible(false);

        gridLayout_2->addWidget(progressBar, 2, 1, 1, 1);

        labelTop = new QLabel(controls);
        labelTop->setObjectName(QStringLiteral("labelTop"));

        gridLayout_2->addWidget(labelTop, 2, 0, 1, 1);


        gridLayout->addWidget(controls, 0, 0, 1, 1);

        glLayout = new QGridLayout();
        glLayout->setObjectName(QStringLiteral("glLayout"));

        gridLayout->addLayout(glLayout, 2, 0, 1, 1);

        line = new QFrame(centralwidget);
        line->setObjectName(QStringLiteral("line"));
        sizePolicy1.setHeightForWidth(line->sizePolicy().hasHeightForWidth());
        line->setSizePolicy(sizePolicy1);
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line, 1, 0, 1, 1);

        widget_2 = new QWidget(centralwidget);
        widget_2->setObjectName(QStringLiteral("widget_2"));
        QSizePolicy sizePolicy3(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(widget_2->sizePolicy().hasHeightForWidth());
        widget_2->setSizePolicy(sizePolicy3);
        widget_2->setMinimumSize(QSize(200, 0));
        widget_2->setMaximumSize(QSize(200, 16777215));
        verticalLayout = new QVBoxLayout(widget_2);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        groupBox_5 = new QGroupBox(widget_2);
        groupBox_5->setObjectName(QStringLiteral("groupBox_5"));
        label_9 = new QLabel(groupBox_5);
        label_9->setObjectName(QStringLiteral("label_9"));
        label_9->setGeometry(QRect(10, 60, 91, 16));
        usedMemLCD = new QLCDNumber(groupBox_5);
        usedMemLCD->setObjectName(QStringLiteral("usedMemLCD"));
        usedMemLCD->setGeometry(QRect(110, 60, 64, 23));
        label_10 = new QLabel(groupBox_5);
        label_10->setObjectName(QStringLiteral("label_10"));
        label_10->setGeometry(QRect(10, 20, 51, 21));
        label_11 = new QLabel(groupBox_5);
        label_11->setObjectName(QStringLiteral("label_11"));
        label_11->setGeometry(QRect(10, 20, 91, 21));
        memSizeLCD = new QLCDNumber(groupBox_5);
        memSizeLCD->setObjectName(QStringLiteral("memSizeLCD"));
        memSizeLCD->setGeometry(QRect(110, 20, 64, 23));
        memSizeLCD->setSegmentStyle(QLCDNumber::Filled);
        fpsLCD = new QLCDNumber(groupBox_5);
        fpsLCD->setObjectName(QStringLiteral("fpsLCD"));
        fpsLCD->setGeometry(QRect(110, 90, 64, 23));
        label_12 = new QLabel(groupBox_5);
        label_12->setObjectName(QStringLiteral("label_12"));
        label_12->setGeometry(QRect(10, 90, 91, 16));

        verticalLayout->addWidget(groupBox_5);


        gridLayout->addWidget(widget_2, 2, 1, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 1112, 21));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        MainWindow->setMenuBar(menubar);

        menubar->addAction(menuFile->menuAction());
        menuFile->addAction(actionOpen);
        menuFile->addSeparator();
        menuFile->addAction(actionClose);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Visualisierung 1", 0));
        actionOpen->setText(QApplication::translate("MainWindow", "Open ...", 0));
        actionClose->setText(QApplication::translate("MainWindow", "Close", 0));
        labelTop->setText(QApplication::translate("MainWindow", "No data loaded", 0));
        groupBox_5->setTitle(QApplication::translate("MainWindow", "GPU", 0));
        label_9->setText(QApplication::translate("MainWindow", "Used Memory (MB)", 0));
        label_10->setText(QString());
        label_11->setText(QApplication::translate("MainWindow", "Total Memory (MB)", 0));
        label_12->setText(QApplication::translate("MainWindow", "FPS", 0));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
