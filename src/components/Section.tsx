import React, { ReactNode } from "react";
import { Box, useColorModeValue } from "@chakra-ui/react";

type SectionProps = {
  children: ReactNode;
  variant?: "light" | "dark";
};

const Section: React.FC<SectionProps> = ({ children, variant = "light" }) => {
  // Define color schemes for light and dark variants
  const bgColorLight = useColorModeValue("#F8F8F8", "#1F1F1F");
  const bgColorDark = useColorModeValue("#FFFFFF", "#121212");
  const textColorLight = useColorModeValue("black", "white");
  const textColorDark = useColorModeValue("black", "white");

  // Choose color based on variant prop
  const bgColor = variant === "dark" ? bgColorDark : bgColorLight;
  const textColor = variant === "dark" ? textColorDark : textColorLight;

  return (
    <Box bg={bgColor} color={textColor} p={4} width="100%" position="relative">
      <Box
        minH="50px"
        width="100%"
        px="15px"
        mx="auto"
        my="50px"
        maxW={{
          base: "100%",    // Full width on small screens
          md: "80%",       // 80% width on medium screens
          lg: "70%",       // 70% width on larger screens
          xl: "60%",       // 60% width on extra-large screens
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default Section;
