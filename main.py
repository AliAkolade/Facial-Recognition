from kivy.app import App

from kivy.uix.anchorlayout import AnchorLayout

class xLayout(AnchorLayout):
    pass

class gui(App):
    def build(self):
        # returning the instance of StackLayout class
        return xLayout()


if __name__ == '__main__':
    gui().run()
