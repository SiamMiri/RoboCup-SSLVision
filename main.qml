import QtQuick 2.6
import QtQuick.Controls 2.15

ApplicationWindow {
    visible: true
    width: 650
    height: 500
    title: "SSl"

    
    Button{
        id: btnReceive
        x: 30
        y: 29
        width: 200
        height: 40
        background: Rectangle {
            color: "Gray"
        }
        Text {
            id: txtBtnReceive
            x: 5
            y: 15
            text: qsTr("Read UDP Buffer")
            color: "#ffffff"
        }
        onClicked : showScreen.color  = "Yellow"
    }

    Rectangle {
        id: btnStopReceive
        x: 415
        y: 29
        width: 200
        height: 40
        color: "black"
        Text {
            id: txtBtnStopReceive
            x: 5
            y: 15
            text: qsTr("Stop Reading UDP Buffer")
            color: "#ffffff"

        }
        MouseArea {
            id: mouseAreaBtnStopReceive
            x: 0
            y: 0
            width: 200
            height: 40
            onClicked: showScreen.color  = "Red"
        }
    }

    Rectangle {
        id: showScreen
        x: 30
        y: 110
        width: 585
        height: 345
        color: "green"

        Rectangle {
            id: rectMiddleLine
            x: 290
            y: 0
            width: 5
            height: 345
            color: "#ffffff"
        }

        Rectangle {
            id: rectGoal1
            x: 0
            y: 86.25
            width: 25
            height: 5
            color: "#ffffff"
        }
        Rectangle {
            id: rectGoal2
            x: 0
            y: 258.75
            width: 25
            height: 5
            color: "#ffffff"
        }

        Rectangle {
            id: rectGoal3
            x: 25
            y: 86.25
            width: 5
            height: 177.5
            color: "#ffffff"
        }
    }
}