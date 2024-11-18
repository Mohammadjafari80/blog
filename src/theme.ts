import { extendTheme, ThemeConfig } from "@chakra-ui/react";
import { mode } from "@chakra-ui/theme-tools";

const config: ThemeConfig = {
  initialColorMode: "light",
};

const theme = extendTheme({
  config,
  styles: {
    global: (props: any) => ({
      body: {
        bg: mode("#FFFFFF", "#121212")(props),
      },
    }),
  },
  fonts: {
    heading: "'Soleil', 'Lato', sans-serif",
    body: "'Soleil', 'Lato', sans-serif",
  },
});

export default theme;
