import {
  HStack,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  IconButton,
  useBreakpointValue,
  Portal,
  Spacer,
  useColorMode,
  Tabs,
  TabList,
  Tab,
  TabIndicator,
} from "@chakra-ui/react";
import { HamburgerIcon } from "@chakra-ui/icons";
import ColorModeSwitch from "./ColorModeSwitch";
import { Link as RouterLink } from "react-router-dom"; // Import RouterLink for internal navigation

const NavBar = () => {
  const padding = 10;
  const isMobile = useBreakpointValue({ base: true, md: false });
  const { colorMode } = useColorMode();
  const bgColor = colorMode === "dark" ? "#121212" : "#FFFFFF";

  const navItems = [
    { name: "Blog", path: "/" }, // Adjusted path for HashRouter
    { name: "Main Site", path: "https://mohammadjafari80.github.io" }, // External URL
  ];

  const renderMenuItems = () => (
    <Tabs variant="unstyled">
      <TabList>
        {navItems.map((item) => (
          item.path.startsWith("http") ? (
            // External link
            <a key={item.name} href={item.path} target="_blank" rel="noopener noreferrer">
              <Tab fontSize={15} fontWeight="bold" aria-label={`${item.name} section`}>
                {item.name}
              </Tab>
            </a>
          ) : (
            // Internal link
            <RouterLink key={item.name} to={item.path}>
              <Tab fontSize={15} fontWeight="bold" aria-label={`${item.name} section`}>
                {item.name}
              </Tab>
            </RouterLink>
          )
        ))}
      </TabList>
      <TabIndicator mt="-40px" height="2px" bg="#d05a45" borderRadius="5px" />
    </Tabs>
  );

  const renderDropdownMenu = () => (
    <Menu>
      <MenuButton
        as={IconButton}
        icon={<HamburgerIcon />}
        mr={5}
        aria-label="Navigation menu"
      />
      <Portal>
        <MenuList width="100%" position="fixed" top="0" left="0" bg={bgColor}>
          {navItems.map((item) =>
            item.path.startsWith("http") ? (
              // External link
              <a
                key={item.name}
                href={item.path}
                target="_blank"
                rel="noopener noreferrer"
              >
                <MenuItem bg={bgColor}>{item.name}</MenuItem>
              </a>
            ) : (
              // Internal link
              <RouterLink key={item.name} to={item.path}>
                <MenuItem bg={bgColor}>{item.name}</MenuItem>
              </RouterLink>
            )
          )}
          <MenuItem bg={bgColor}>
            <ColorModeSwitch />
          </MenuItem>
        </MenuList>
      </Portal>
    </Menu>
  );

  return (
    <HStack
      m={padding}
      bgColor={bgColor}
      alignItems="baseline"
      w="100%"
      pl="15px"
      pr="15px"
      ml="auto"
      mr="auto"
      mb="0px"
      maxWidth="95%"
      css={{
        "@media (min-width: 1200px)": {
          maxWidth: "60%",
        },
      }}
    >
      {isMobile ? renderDropdownMenu() : renderMenuItems()}
      {!isMobile && (
        <>
          <Spacer />
          <ColorModeSwitch />
        </>
      )}
    </HStack>
  );
};

export default NavBar;
