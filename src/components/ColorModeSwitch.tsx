import { HStack, IconButton, useColorMode } from "@chakra-ui/react";
import { SunIcon, MoonIcon } from "@chakra-ui/icons";

const ColorModeSwitch = () => {
  const { toggleColorMode, colorMode } = useColorMode();
  return (
    <HStack>
      <IconButton
        onClick={toggleColorMode}
        icon={colorMode === "light" ? <MoonIcon /> : <SunIcon />}
        aria-label={`Switch to ${
          colorMode === "light" ? "dark" : "light"
        } mode`}
      />
    </HStack>
  );
};

export default ColorModeSwitch;
