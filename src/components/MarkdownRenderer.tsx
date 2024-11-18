import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Helmet } from 'react-helmet-async';
import { Highlight, themes } from 'prism-react-renderer';
import {
  Heading,
  Text,
  Image,
  Link,
  Code,
  ListItem,
  UnorderedList,
  OrderedList,
  useColorModeValue,
  Box,
  Divider,
} from '@chakra-ui/react';

interface MarkdownRendererProps {
  content: string;
  dir?: 'ltr' | 'rtl';
  meta?: {
    title?: string;
    description?: string;
    image?: string;
    url?: string;
  };
}

const colors = {
  blue: "#3f6ee7",
  green: "#4a9c80",
  red: "#d05a45"
};


// const [gradientColors] = useState([
//     "#d05a45", // Color 1
//     "#e4aa42", // Color 2
//     "#3f6ee7", // Color 3
//     "#4a9c80", // Color 4
//   ]);

const CodeBlock = ({ children, className }: any) => {
    const language = className ? className.replace(/language-/, '') : 'javascript';
    
    return (
      <Highlight
        theme={themes.nightOwl}
        code={children.trim()}
        language={language as any}
      >
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <Box
            as="pre"
            p={4}
            my={4}
            borderRadius="lg"
            bg="gray.900"
            overflowX="auto"
            fontSize="sm"
            border="1px solid"
            borderColor={`${colors.blue}22`}
            position="relative"
          >
            {/* Language badge */}
            <Box
              position="absolute"
              top={2}
              right={2}
              px={2}
              py={1}
              fontSize="xs"
              color={colors.green}
              bg={`${colors.green}22`}
              borderRadius="md"
              textTransform="uppercase"
              letterSpacing="wide"
            >
              {language}
            </Box>
            <Box fontFamily="mono">
              {tokens.map((line, i) => (
                <Box {...getLineProps({ line, key: i })} display="table-row">
                  {/* Line number */}
                  <Box
                    display="table-cell"
                    pr={4}
                    opacity={0.5}
                    userSelect="none"
                    color={colors.green}
                  >
                    {i + 1}
                  </Box>
                  {/* Code content */}
                  <Box display="table-cell">
                    {line.map((token, key) => (
                      <span {...getTokenProps({ token, key })} />
                    ))}
                  </Box>
                </Box>
              ))}
            </Box>
          </Box>
        )}
      </Highlight>
    );
  };

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ 
  content, 
  dir = 'ltr',
  meta 
}) => {
  const linkColor = useColorModeValue(colors.blue, `${colors.blue}dd`);
  const codeBgColor = useColorModeValue('gray.50', 'gray.800');

  const components = {
    h1: (props: any) => (
      <Heading 
        as="h1" 
        size="2xl" 
        my={6} 
        color={`${colors.blue}ff`}  // 100% opacity
        fontWeight="bold"
        dir={dir}
        {...props} 
      />
    ),
    h2: (props: any) => (
      <Heading 
        as="h2" 
        size="xl" 
        my={4} 
        color={`${colors.blue}cc`}  // 80% opacity
        fontWeight="bold"
        dir={dir}
        {...props} 
      />
    ),
    h3: (props: any) => (
      <Heading 
        as="h3" 
        size="lg" 
        my={3} 
        color={`${colors.green}cc`}  // 80% opacity
        fontWeight="semibold"
        dir={dir}
        {...props} 
      />
    ),
    h4: (props: any) => (
      <Heading 
        as="h4" 
        size="md" 
        my={2} 
        color={`${colors.green}99`}  // 60% opacity
        fontWeight="semibold"
        dir={dir}
        {...props} 
      />
    ),
    p: (props: any) => (
      <Text 
        my={3} 
        dir={dir}
        color={useColorModeValue('gray.800', 'gray.200')}
        lineHeight="tall"
        {...props} 
      />
    ),
    a: (props: any) => (
      <Link 
        // color={linkColor} 
        textDecoration="none"
        borderBottom="2px solid"
        // borderColor={`${colors.blue}44`}
        // _hover={{ 
        //   color: colors.green,
        //   borderColor: colors.green,
        // }}
        transition="all 0.2s ease"
        {...props} 
        isExternal 
      />
    ),
    img: (props: any) => (
      <Box
        position="relative"
        my={4}
      >
        <Image 
          borderRadius="lg"
          loading="lazy"
        //   border="1px solid"
        //   borderColor={`${colors.green}33`}
        //   _hover={{
        //     borderColor: `${colors.green}66`
        //   }}
        w={'100%'}
        align={'center'}
        m={'auto'}
          transition="border-color 0.2s ease"
          {...props} 
        />
      </Box>
    ),
    code: (props: any) => {
      // If there's a language specified, use CodeBlock
      if (props.className) {
        return <CodeBlock {...props} />;
      }
      // Otherwise render inline code
      return (
        <Code
          display="inline-block"
          px={2}
          py={1}
          borderRadius="md"
          bg={codeBgColor}
          color={colors.red}
          fontSize="sm"
          fontFamily="mono"
          dir="ltr"
          border="1px solid"
          borderColor={`${colors.red}22`}
          {...props}
        />
      );
    },
    li: (props: any) => (
      <ListItem 
        fontSize="md" 
        my={2} 
        dir={dir}
        transition="color 0.2s ease"
        {...props} 
      />
    ),
    ul: (props: any) => (
      <UnorderedList 
        spacing={2} 
        pl={dir === 'rtl' ? 0 : 6}
        pr={dir === 'rtl' ? 6 : 0}
        my={4}
        sx={{
          '& li::marker': {
            color: `${colors.green}aa`,
          }
        }}
        {...props} 
      />
    ),
    ol: (props: any) => (
      <OrderedList 
        spacing={2} 
        pl={dir === 'rtl' ? 0 : 6}
        pr={dir === 'rtl' ? 6 : 0}
        my={4}
        {...props} 
      />
    ),
    blockquote: (props: any) => (
      <Box
        pl={4}
        py={2}
        my={4}
        borderLeftWidth={4}
        borderLeftColor={colors.green}
        bg={`${colors.green}11`}
        borderRadius="md"
        fontStyle="italic"
        dir={dir}
        _hover={{
          borderLeftColor: colors.green,
          bg: `${colors.green}11`
        }}
        transition="all 0.2s ease"
        {...props}
      />
    ),
    hr: () => (
      <Divider 
        my={6}
        borderColor={`${colors.blue}33`}
        _hover={{
          borderColor: `${colors.green}66`
        }}
        transition="border-color 0.2s ease"
      />
    ),
    table: (props: any) => (
      <Box
        as="table"
        width="full"
        my={4}
        borderWidth="1px"
        borderColor={`${colors.green}33`}
        borderRadius="lg"
        overflow="hidden"
        {...props}
      />
    ),
    th: (props: any) => (
      <Box
        as="th"
        p={2}
        bg={`${colors.blue}11`}
        borderBottomWidth="1px"
        borderColor={`${colors.blue}33`}
        color={colors.blue}
        fontWeight="semibold"
        textAlign="left"
        {...props}
      />
    ),
    td: (props: any) => (
      <Box
        as="td"
        p={2}
        borderBottomWidth="1px"
        borderColor={`${colors.green}22`}
        _hover={{
          bg: `${colors.green}05`
        }}
        transition="background-color 0.2s ease"
        {...props}
      />
    ),
  };

  return (
    <>
      {meta && <Helmet>...</Helmet>}
      <Box 
        dir={dir}
        sx={{
          '& > :first-of-type': { mt: 0 },
          '& > :last-child': { mb: 0 }
        }}
      >
        <ReactMarkdown components={components} remarkPlugins={[remarkGfm]}>
          {content}
        </ReactMarkdown>
      </Box>
    </>
  );
};

export default MarkdownRenderer;